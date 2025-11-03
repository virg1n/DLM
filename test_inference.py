from model import Transformer, ModelConfig

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

checkpoint_path = 'model_testing/model.checkpoint.0.pt'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = ModelConfig(
        vocab_size = effective_vocab,

        num_dims = 512,
        num_heads = 16,
        num_kv_heads = 16,
        num_layers = 32,
        ffn_hidden_dims = 512 * 4,

        rmsnorm_eps = 1e-6,
        rope_theta = 1e5,

        context_len = 1536,
        use_flash = False,

        mask_token_id = int(tokenizer.mask_token_id),
)

model = Transformer(config)
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

state_dict = checkpoint['model']
new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith("_orig_mod."):
        new_state_dict[k[len("_orig_mod."):]] = v 
    else:
        new_state_dict[k] = v

model.load_state_dict(new_state_dict, strict=False)

model.to(device)
model.eval()

prompt_text = "city, seat of Delaware county,"
prompt_tok = tokenizer.encode(prompt_text, return_tensors="pt").to(device)

given_len, gen_len = len(prompt_tok), 1531
mask_id = model.config.mask_token_id

prompt = torch.cat([prompt_tok.to(device), torch.tensor([mask_id] * gen_len).unsqueeze(0).to(device)], dim=-1)

x_masked = prompt.clone().to(device)
mask = torch.zeros_like(prompt, dtype=torch.bool).to(device)
mask[:, 5:] = True
x_masked[mask] = int(mask_id)

with torch.autocast(device_type=device.type, dtype=getattr(torch, "bfloat16")):
    completed = model.generate(
        x_masked, steps=8, temperature=1.0, mask=mask, sample=False, top_k=10
    )

print("Generated text:\n", tokenizer.decode(completed[0].tolist()))