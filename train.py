from model import Transformer, ModelConfig
from trainer import Trainer, TrainerConfig, DataLoader

from transformers import AutoTokenizer
import torch

torch.set_float32_matmul_precision('high')
torch.cuda.empty_cache()

tokenizer_id = "HuggingFaceTB/SmolLM-360M"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

orig_vocab = tokenizer.vocab_size

if tokenizer.mask_token is None:
    tokenizer.add_special_tokens({"mask_token": "<mask>"})

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "<pad>"})

effective_vocab = len(tokenizer)

checkpoint_path = 'model_testing/model.checkpoint.600.pt'
continue_train = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_config = TrainerConfig(
    vocab_size = effective_vocab,
    num_epochs = 1,

    use_ddp = False,
    clean_cuda_cache = True,
    use_compile = False,
    use_dtype = "bfloat16",

    seed = 1338,
    max_seq_len = 1536,
    batch_size = 2,
    accumulation_steps = 16,

    weight_decay = 0.1,
    warmup_ratio = 0.1,
    learning_rate = 7e-4,
    betas = (0.90, 0.97),

    val_ratio = 0.005,
    steps_for_eval = 5,
    eval_interval = 30,

    checkpoints_frequency = 300, 
    path_to_checkpoints = "./model_testing",

    tokenized_dataset_path = "fwe-10BT", 
    eval_log_file = "logs/eval.txt", 
)

config = ModelConfig(
        vocab_size = effective_vocab,

        num_dims = 512,
        num_heads = 16,
        num_kv_heads = 4,
        num_layers = 32,
        ffn_hidden_dims = 512 * 4,

        rmsnorm_eps = 1e-6,
        rope_theta = 1e5,

        context_len = 1536,
        use_flash = True,

        mask_token_id = int(tokenizer.mask_token_id),
)

model = Transformer(config)
if continue_train:
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

    state_dict = checkpoint['model']
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            new_state_dict[k[len("_orig_mod."):]] = v 
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=False)

data_loader = DataLoader(train_config)


print("--- Loading 4 fixed batches ---")
fixed_batches = []
for i in range(4):
    batch = data_loader.next_batch(split="train")
    fixed_batches.append(batch)
    print(f"--- Batch {i+1} First 10 Words ---")
    first_sequence_tokens = batch[0][:30]
    text = tokenizer.decode(first_sequence_tokens)
    print(f"'{text}'")


probe_max = 0
for b in fixed_batches:
    probe_max = max(probe_max, int(b.max().item()))

req_vocab = max(
    effective_vocab,
    probe_max + 1,
    int(tokenizer.mask_token_id) + 1,
    (int(tokenizer.pad_token_id) + 1) if tokenizer.pad_token_id is not None else 0,
)

def resize_vocab_(model, new_vocab):
    old_vocab = model.tokens_embedding.num_embeddings
    if new_vocab <= old_vocab:
        return
    with torch.no_grad():
        E = model.tokens_embedding.weight.data
        add = new_vocab - old_vocab
        pad = E.mean(dim=0, keepdim=True).repeat(add, 1)
        new_E = torch.cat([E, pad], dim=0)
        model.tokens_embedding = torch.nn.Embedding.from_pretrained(new_E, freeze=False)
        model.ll_head = torch.nn.Linear(model.num_dims, new_vocab, bias=False)
        model.ll_head.weight = model.tokens_embedding.weight  # re-tie
        model.config.vocab_size = new_vocab
        model.vocab_size = new_vocab

resize_vocab_(model, req_vocab)


class FixedBatchLoader:
    def __init__(self, train_batches, val_loader, config):
        self.train_batches = train_batches 
        self.val_loader = val_loader
        self.config = config
        self.train_idx = 0
        self.num_train_batches = len(train_batches)
        print(f"Created FixedBatchLoader with {self.num_train_batches} training batches.")

        self._num_tokens = self.num_train_batches * self.config.batch_size * self.config.max_seq_len

    def next_batch(self, split="train"):
        if split == "train":
            batch = self.train_batches[self.train_idx]
            self.train_idx = (self.train_idx + 1) % self.num_train_batches
            return batch
        else:
            return self.val_loader.next_batch(split="val")

    @property
    def num_tokens(self):
        return self._num_tokens


    def num_train_steps(self):

        return self.num_train_batches
    
mock_loader = FixedBatchLoader(fixed_batches, data_loader, train_config)

print("Initializing Trainer with (train_config, model, tokenizer)...")
trainer = Trainer(train_config, model, tokenizer)

print("len(tokenizer)=", len(tokenizer),
      "base vocab_size=", tokenizer.vocab_size,
      "mask_id=", tokenizer.mask_token_id,
      "pad_id=", tokenizer.pad_token_id)
print("model V=", model.tokens_embedding.num_embeddings)


print("Starting training with custom FixedBatchLoader...")
trainer.train(data_loader)

print("\n--- Training complete. Starting Generation ---")

model.eval()

prompt_tokens = fixed_batches[0][0][:5].unsqueeze(0).to(device)  # (1, P)
gen_len = 100

mask_id = model.config.mask_token_id
cont = torch.full((1, gen_len), fill_value=mask_id, dtype=prompt_tokens.dtype, device=prompt_tokens.device)

x_init = torch.cat([prompt_tokens, cont], dim=1)

mask = torch.zeros_like(x_init, dtype=torch.bool)
mask[:, prompt_tokens.size(1):] = True

steps = 60

with torch.no_grad():
    completed = model.generate(x_init, steps=steps, temperature=1, mask=mask, sample=False, top_k=50)

generated_text = tokenizer.decode(completed[0].tolist(), skip_special_tokens=True)
print(f"Prompt: '{tokenizer.decode(prompt_tokens[0].tolist(), skip_special_tokens=True)}'")
print("Generated text:\n", generated_text)

