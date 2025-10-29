import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from dataclasses import dataclass
from huggingface_hub import PyTorchModelHubMixin
from typing import Optional

from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist


@dataclass
class ModelConfig:
    vocab_size: int

    num_dims: int                       # number of dimensions
    num_heads: int                      # number of query heads
    num_kv_heads: int                   # number of key/value heads
    num_layers: int                     # total transformer layers
    ffn_hidden_dims: int                # hidden dimension for FFN

    context_len: int                    # maximum context length
    use_flash: bool                     # use Flash Attention
    
    rmsnorm_eps: float = 1e-6
    rope_theta: float = 1e5

    mask_token_id: int = 0

    ffn_dim_multiplier: Optional[int] = None    # optional multiplier to compute ffn_hidden_dims


# Helper function for RoPE
def repeat_kv(vct: torch.Tensor, n_times: int):
    c_batch_size, c_context_len, num_kv_heads, c_dim = vct.shape
    if n_times == 1:
        return vct
    else:
        return (
            vct[:, :, :, None, :]
            .expand(c_batch_size, c_context_len, num_kv_heads, n_times, c_dim)
            .reshape(c_batch_size, c_context_len, num_kv_heads * n_times, c_dim)
        )


class Rotary(nn.Module):
    def __init__(self, config):
        super(Rotary, self).__init__()

        inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, config.num_dims // config.num_heads, 2).float() / (config.num_dims // config.num_heads)))
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        self.seq_len_saved = None
        self.cos_saved = None
        self.sin_saved = None

    def forward(self, x, seq_dim=1):
        seq_len = x.size(seq_dim)
        # Only recompute the cosine and sine matrices if the sequence length has changed.
        if seq_len != self.seq_len_saved:
            self.seq_len_saved = seq_len
            pos = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
            # Compute the outer product between positions and inverse frequencies.
            freqs = torch.einsum("i,j->ij", pos, self.inv_freq) # (seq_len, inv_freq.shape[0])
            # Duplicate the freqs along the last dimension to create pairs.
            emb = torch.cat((freqs, freqs), dim=-1)
            self.cos_saved = emb.cos()
            self.sin_saved = emb.sin()

        return self.cos_saved, self.sin_saved


class RMSNorm(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.g = nn.Parameter(torch.ones(config.num_dims))
        self.eps = config.rmsnorm_eps
    
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self.g * self._norm(x.float()).type_as(x)
    

class GroupedQueryAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.use_flash = config.use_flash

        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_heads if config.num_kv_heads is None else config.num_kv_heads

        self.num_rep = self.num_heads // self.num_kv_heads
        self.head_dim = config.num_dims // self.num_heads

        self.wq = nn.Linear(config.num_dims, config.num_dims, bias=False)
        nn.init.normal_(self.wq.weight, mean=0, std=1/math.sqrt(config.num_dims))
        self.wk = nn.Linear(config.num_dims, self.num_kv_heads * self.head_dim, bias=False)
        nn.init.normal_(self.wk.weight, mean=0, std=1/math.sqrt(config.num_dims))
        self.wv = nn.Linear(config.num_dims, self.num_kv_heads * self.head_dim, bias=False)
        nn.init.normal_(self.wv.weight, mean=0, std=1/math.sqrt(config.num_dims))
        
        self.wo = nn.Linear(config.num_dims, config.num_dims, bias=False)

        self.cache_k = None
        self.cache_v = None


    def rotate_half(self, x):
        half = x.shape[-1] // 2
        first_half, second_half  = x[..., :half], x[..., half:]
        return torch.cat([-second_half, first_half], dim=-1)


    def apply_rotary_pos(self, q, k, cos, sin):
        q_rot = q * cos + self.rotate_half(q) * sin
        k_rot = k * cos + self.rotate_half(k) * sin
        return q_rot, k_rot
    

    def forward(self, x, cos, sin, start_pos = 0):
        c_batch_size, c_context_len, c_dim = x.shape # c_context_len = 1


        # Non-cache branch (process the entire sequence normally)
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        q = q.view(c_batch_size, c_context_len, self.num_heads, self.head_dim).transpose(1, 2)      # B, qh, T, hs
        k = k.view(c_batch_size, c_context_len, self.num_kv_heads, self.head_dim).transpose(1, 2)   # B, kh, T, hs
        v = v.view(c_batch_size, c_context_len, self.num_kv_heads, self.head_dim).transpose(1, 2)   # B, vh, T, hs

        queries, keys = self.apply_rotary_pos(q, k, cos, sin)
        # keys = repeat_kv(keys, self.num_rep)
        # values = repeat_kv(v, self.num_rep)

        if self.use_flash:
            output = F.scaled_dot_product_attention(queries, keys, v, is_causal=False, enable_gqa=True)
            
        else: # Calculate Grouped Query Attention manually
            values = repeat_kv(v, self.num_rep)
            keys = repeat_kv(keys, self.num_rep)
            attention = torch.matmul(queries, keys.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))

            attention = F.softmax(attention, dim=-1).type_as(queries)
            output = torch.matmul(attention, values)

        output = output.transpose(2, 1).contiguous().view(c_batch_size, c_context_len, c_dim)
        return self.wo(output)


class FeedForward(nn.Module):
    """
    Default Feed Forward Layer.
    """
    def __init__(self, config):
        super().__init__()

        self.hidden_dim = config.ffn_hidden_dims

        self.w1 = nn.Linear(config.num_dims, self.hidden_dim, bias=False)
        self.w2 = nn.Linear(self.hidden_dim, config.num_dims, bias=False)
        self.w3 = nn.Linear(config.num_dims, self.hidden_dim, bias=False)
        self.act = nn.SiLU()
    def forward(self, x: torch.Tensor):
        return self.w2(self.act(self.w1(x)) * self.w3(x))

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.attention = GroupedQueryAttention(config)

        self.ffn = FeedForward(config)


        self.norm_attention = torch.nn.modules.normalization.RMSNorm(config.num_dims, config.rmsnorm_eps) # you also can use RMSNorm(config)
        self.norm_ffn = torch.nn.modules.normalization.RMSNorm(config.num_dims, config.rmsnorm_eps) # you also can use RMSNorm(config)

    def forward(self, x, cos, sin, start_pos):
        x = x + self.attention(
            self.norm_attention(x), 
            cos, sin, start_pos
            )
        
        ffn_out = self.ffn(
            self.norm_ffn(x)
            )
        x = x + ffn_out
        return x
    

class Transformer(nn.Module, PyTorchModelHubMixin): # extending PyTorchModelHubMixin for save weights as safetensors
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config

        self.vocab_size = config.vocab_size
        self.num_dims = config.num_dims
        self.num_heads = config.num_heads
        self.context_len = config.context_len

        self.num_layers = config.num_layers
        self.rotary_emb = Rotary(config)
        
        self.tokens_embedding = nn.Embedding(self.vocab_size, self.num_dims)

        self.blocks = nn.ModuleList()
        for _ in range(self.num_layers):
            self.blocks.append(Block(config))

        self.norm = torch.nn.modules.normalization.RMSNorm(config.num_dims, config.rmsnorm_eps) # you also can use RMSNorm(config)
        self.ll_head = nn.Linear(self.num_dims, self.vocab_size, bias=False)

        self.ll_head.weight = self.tokens_embedding.weight

        

        # self.tokens_embedding.weight = self.ll_head.weight
        # torch.nn.init.normal_(self.ll_head.weight, mean=0.0, std=0.02)
        # torch.nn.init.normal_(self.tokens_embedding.weight, mean=0.0, std=0.02)




    
    def forward(
            self,
            x_masked: torch.LongTensor,          # (B, T)  tokens after masking in the train loop
            targets: torch.LongTensor = None,    # (B, T)  targets
            mask: torch.BoolTensor = None,       # (B, T)  True where xi_t == [MASK]
            t: Optional[torch.Tensor] = None,    # (B,)    sampled mask ratios per example
            ):
        xmin = int(x_masked.min().item()); xmax = int(x_masked.max().item())
        V = self.tokens_embedding.num_embeddings
        assert 0 <= xmin and xmax < V, f"input id out of range: min={xmin}, max={xmax}, V={V}"

        x = self.tokens_embedding(x_masked)
        cos, sin = self.rotary_emb(x, seq_dim=1)

        for block in self.blocks:
            x = block(x, cos, sin, start_pos=0)
        
        x = self.norm(x)
        logits = self.ll_head(x)

        c_batch_size, c_context_len, c_dim = logits.shape

        
        
        if targets is None:
            return logits, None
        logits = logits.view(c_batch_size*c_context_len, c_dim)
        targets = targets.view(c_batch_size*c_context_len)
        assert targets.dtype == torch.long, f"targets dtype {targets.dtype} must be long"
        mn = int(targets.min().item()); mx = int(targets.max().item())
        assert mn >= 0, f"negative target id: min={mn}"
        assert mx < c_dim, f"target id {mx} >= vocab dim V={c_dim}"
        ce_value = F.cross_entropy(logits, targets, reduction='none').view(c_batch_size, c_context_len)
        
        masked_loss = ce_value * mask.float() # zero loss where mask == False

        loss_per_seq = masked_loss.sum(dim=1) / (t.clamp_min(1e-8) * float(c_context_len))  # (B,)
        loss = loss_per_seq.mean() 

        return logits, loss


    @torch.no_grad()
    def generate(self,
             x_init: torch.LongTensor,
             steps: int = 8,
             temperature: float = 1.0,
             top_k: int = 0,
             mask: torch.BoolTensor = None,
             sample: bool = False):
        x = x_init.clone()
        if mask is None:
            mask = (x == self.config.mask_token_id)
        else:
            mask = mask.bool()

        # uniform timetable t from 1.0 -> 0.0
        tgrid = torch.linspace(1.0, 0.0, steps + 1, device=x.device)

        for sidx in range(steps):
            t = float(tgrid[sidx].item())
            s = float(tgrid[sidx + 1].item())

            logits, _ = self.forward(x_masked=x)
            if temperature != 1.0:
                logits = logits / max(temperature, 1e-6)
            if top_k > 0:
                topv, _ = torch.topk(logits, top_k, dim=-1)
                threshold = topv[..., -1].unsqueeze(-1)
                logits = logits.masked_fill(logits < threshold, float("-inf"))

            probs = F.softmax(logits, dim=-1)
            pred = (torch.multinomial(probs.view(-1, probs.size(-1)), 1).view_as(x)
                    if sample else probs.argmax(dim=-1))

            # fill only where currently masked
            x = torch.where(mask, pred, x)

            # final step — stop before remasking
            if sidx == steps - 1:
                break

            conf, _ = probs.max(dim=-1)                 # (B, T)
            conf_masked = conf.masked_fill(~mask, 1.0)  # 1.0 so they won't be picked

            B = x.size(0)
            new_mask = torch.zeros_like(mask)

            mt = mask.sum(dim=1)  # how many are masked now
            keep_counts = torch.ceil(mt.float() * (s / max(t, 1e-6))).to(torch.int64)

            for b in range(B):
                k = keep_counts[b].item()
                if k <= 0 or mt[b].item() == 0:
                    continue
                _, idxs = torch.topk(conf_masked[b], k, largest=False)
                keep = torch.zeros_like(mask[b])
                keep[idxs] = True
                new_mask[b] = keep & mask[b]

            mask = new_mask

        return x

    

def main():
    pass
    # config = ModelConfig(
    #     # device = 'cuda' if torch.cuda.is_available() else 'cpu',
    #     vocab_size = 50304,

    #     num_dims = 1024,
    #     num_heads = 16,
    #     num_kv_heads = 4,
    #     num_layers = 16,
    #     ffn_hidden_dims = 1024 * 4,

    #     rmsnorm_eps = 1e-6,
    #     rope_theta = 1e5,

    #     context_len = 1024,
        
    #     use_flash = False,
    # )

    
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # SEED = 1337

    # torch.manual_seed(SEED)
    # if device == 'cuda':
    #     torch.cuda.manual_seed(SEED)

    # model = Transformer(config)
    # model = model.to(device)
    # model = torch.compile(model)

    # print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')


if __name__ == "__main__":
    main()


