import time
import math
import os
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Tuple

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from datatrove.utils.dataset import DatatroveFolderDataset

@dataclass
class TrainerConfig:
    vocab_size: int
    num_epochs: int

    use_ddp: bool
    clean_cuda_cache: bool = True
    use_compile: bool = True
    use_dtype: str = "bfloat16"

    seed: int = 1998
    max_seq_len: int = 1024
    batch_size: int = 1
    accumulation_steps: int = 1

    # Optimizer parameters
    weight_decay: float = 0.1
    warmup_ratio: float = 0.01
    learning_rate: float = 1e-3
    betas: Tuple[float, float] = (0.90, 0.95)

    val_ratio: float = 0.005
    steps_for_eval: int = 20
    eval_interval: int = 50

    checkpoints_frequency: int = 500
    path_to_checkpoints: str = "./model_testing"

    tokenized_dataset_path: str = ""
    eval_log_file: str = "logs/eval.txt"


class DataLoader:
    """
    Loads fixed-length token sequences and returns batches of *unshifted* token IDs.
    Masking for diffusion training is performed inside Trainer.step().
    """
    def __init__(self, config, rank=0, world_size=1):
        self.config = config
        self.current_epoch = 0
        self.seed = config.seed
        self.token_size = 2 if config.vocab_size < 65535 else 4
        self.rank = rank

        self.load_dataset(self.seed)
        self.len_dataset = len(self.dataset)

        if rank == 0:
            print(f"{'Total tokens loaded: '} {self.len_dataset * config.max_seq_len:,}")

        self.train_len_dataset = math.ceil((1 - config.val_ratio) * self.len_dataset)
        self.val_len_dataset = self.len_dataset - self.train_len_dataset

        shard_size = self.train_len_dataset // world_size if world_size > 0 else self.train_len_dataset
        self.train_start_idx = rank * shard_size
        self.train_end_idx = self.train_start_idx + shard_size
        self.train_current_idx = self.train_start_idx

        self.val_start_idx = self.train_len_dataset
        self.val_current_idx = self.val_start_idx

    def get_batch(self, current_idx: int, start_idx: int, end_idx: int):
        new_idx = current_idx + self.config.batch_size

        seqs = []
        for idx in range(current_idx, min(new_idx, end_idx)):
            # Unshifted full sequence length max_seq_len
            seqs.append(self.dataset[idx]["input_ids"])
        if not seqs:
            # Safety: if we somehow overrun, restart epoch
            new_idx = start_idx
            self.new_epoch()
            seqs = [self.dataset[new_idx]["input_ids"]]

        x = torch.stack(seqs)  # (B, T)

        if new_idx >= end_idx:
            new_idx = start_idx
            self.new_epoch()

        return x, new_idx

    def next_batch(self, split):
        if split == "train":
            x, self.train_current_idx = self.get_batch(self.train_current_idx, self.train_start_idx, self.train_end_idx)
        else:  # validation
            x, self.val_current_idx = self.get_batch(self.val_current_idx, self.val_start_idx, self.len_dataset)
        return x

    def reset(self, rank: int = 0, world_size: int = 1):
        self.current_epoch = 0
        self.seed = self.config.seed
        self.load_dataset(self.seed)
        self.len_dataset = len(self.dataset)

        self.train_len_dataset = math.ceil((1 - self.config.val_ratio) * self.len_dataset)
        self.val_len_dataset = self.len_dataset - self.train_len_dataset

        shard_size = self.train_len_dataset // world_size if world_size > 0 else self.train_len_dataset
        self.train_start_idx = rank * shard_size
        self.train_end_idx = self.train_start_idx + shard_size
        self.train_current_idx = self.train_start_idx

        self.val_start_idx = self.train_len_dataset
        self.val_current_idx = self.val_start_idx

    def new_epoch(self):
        self.current_epoch += 1
        self.load_dataset(self.seed + self.current_epoch)

    def load_dataset(self, seed: int):
        self.dataset = DatatroveFolderDataset(
            folder_path=self.config.tokenized_dataset_path,
            filename_pattern="**/*.ds", #os.path.join(self.config.tokenized_dataset_path, "**", "*.ds"),
            seq_len=self.config.max_seq_len,
            token_size=self.token_size,
            recursive=True,
            shuffle=True,
            seed=seed + self.rank,
        )

    def num_train_steps(self):
        return math.ceil((self.train_end_idx - self.train_start_idx) / self.config.batch_size)


class Trainer:
    def __init__(self, config, model, tokenizer):
        self.config = config
        self.model = model
        self.num_epochs = config.num_epochs

        self.clean_cuda_cache = config.clean_cuda_cache
        self.dtype = getattr(torch, self.config.use_dtype)

        self.steps_for_eval = config.steps_for_eval
        self.weight_decay = config.weight_decay

        self.tokenizer = tokenizer

        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        n_gpus = torch.cuda.device_count() if self.device.type == "cuda" else 0
        if self.device.type == "cuda":
            torch.cuda.manual_seed(config.seed)

        use_compile = self.config.use_compile and self.device.type == "cuda" and torch.__version__.startswith("2")
        if use_compile:
            self.model = torch.compile(self.model)

        # Mask token id comes from model config; default to 0 (matches your ModelConfig)
        self.mask_token_id = getattr(getattr(self.model, "config", None), "mask_token_id", 0)

        # DDP
        if n_gpus > 1 and config.use_ddp:
            self.ddp = True
            self.ddp_rank = int(os.environ["RANK"])
            self.ddp_local_rank = int(os.environ["LOCAL_RANK"])
            self.ddp_world_size = int(os.environ["WORLD_SIZE"])
            self.device = torch.device(f"cuda:{self.ddp_local_rank}")
            torch.cuda.set_device(self.device)
            self.master_process = self.ddp_rank == 0

            self.model.to(self.device)
            self.model = DDP(self.model, device_ids=[self.ddp_local_rank])
            self.raw_m = model
        else:
            self.ddp = False
            self.ddp_rank = 0
            self.ddp_world_size = 1
            self.master_process = True
            if self.device.type != "cpu":
                self.model.to(self.device)

        if self.master_process:
            print("Device:", self.device)
            print(
                f"Model's trainable params: {sum([p.data.numel() for p in self.model.parameters() if p.requires_grad]) / 1e6:.2f}M"
            )
            print(
                f"Tokens per step: {self.config.batch_size * self.config.max_seq_len * self.ddp_world_size * self.config.accumulation_steps}"
            )
            print(f"use {'torch.compile()' if use_compile else 'eager'}: {use_compile}")


    def _sample_mask(self, x: torch.LongTensor):
        """
        x: (B, T) original tokens
        Returns:
            x_masked: (B, T) tokens where masked positions set to mask_token_id
            targets:  (B, T) original tokens
            mask:     (B, T) bool tensor of masked positions
            t:        (B,)   per-example mask ratios sampled ~ U[0,1]
        """
        B, T = x.shape
        device = x.device

        t = torch.rand(B, device=device)  # (B)
        t = t.clamp(0.1, 0.9)

        probs = t.unsqueeze(1).expand(B, T)
        mask = torch.bernoulli(probs).bool()  # (B, T)

        none_masked = ~mask.any(dim=1)
        if none_masked.any():
            rand_cols = torch.randint(low=0, high=T, size=(none_masked.sum(),), device=device)
            mask[none_masked, :] = False
            mask[none_masked, rand_cols] = True

        x_masked = x.clone()
        x_masked[mask] = self.mask_token_id

        targets = x  
        return x_masked, targets, mask, t

    def step(self, data_loader, accumulation_steps: int, num_tokens: int, split: str = "train"):
        """
        Performs single forward/backward pass with gradient accumulation.
            Returns: (loss (reduced), number_of_processed_tokens)
        """
        x = data_loader.next_batch(split=split)  # (B, T)
        x = x.to(self.device)
        num_tokens += torch.numel(x)

        x_masked, targets, mask, t = self._sample_mask(x)

        with torch.autocast(device_type=self.device.type, dtype=self.dtype):
            _, loss = self.model(
                x_masked=x_masked,
                targets=targets,
                mask=mask,
                t=t,
            )

        loss = loss / accumulation_steps
        loss.backward()
        return loss, num_tokens

    def train(self, data_loader):
        num_steps_per_epoch = math.ceil(data_loader.num_train_steps() / self.config.accumulation_steps)

        # Optimizer & schedulers (AdamW + warmup + cosine decay)
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            betas=self.config.betas,
            weight_decay=self.weight_decay,
            fused=(self.device.type == "cuda"),
        )

        warmup_steps = math.floor(self.config.warmup_ratio * num_steps_per_epoch * self.num_epochs)
        warmup_factor = lambda step: 0.05 + 0.95 * (step / max(warmup_steps, 1))
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_factor)

        cos_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=(num_steps_per_epoch * self.num_epochs) - warmup_steps, eta_min=0.1 * self.config.learning_rate
        )

        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, cos_scheduler], milestones=[warmup_steps]
        )

        last_step = num_steps_per_epoch - 1
        self.model.train()

        for epoch in range(self.num_epochs):
            for step in range(num_steps_per_epoch):
                t0 = time.perf_counter()
                accumulated_loss = 0.0
                num_tokens = 0

                ddp_nosync_ctx = self.model.no_sync() if self.ddp else nullcontext()
                with ddp_nosync_ctx:
                    for _ in range(self.config.accumulation_steps - 1):
                        loss, num_tokens = self.step(
                            data_loader, self.config.accumulation_steps, num_tokens, split="train"
                        )
                        accumulated_loss += loss

                loss, num_tokens = self.step(data_loader, self.config.accumulation_steps, num_tokens, split="train")
                accumulated_loss += loss.detach()

                norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

                t1 = time.perf_counter()

                tokens_per_sec = num_tokens / (t1 - t0) * self.ddp_world_size

                # Logging
                if self.master_process:
                    print(
                        f"Epoch: {epoch} | Step: {step} | loss: {accumulated_loss:.4f} | norm: {norm:.4f} | "
                        f"lr: {scheduler.get_last_lr()[0]:.6e} | tok/s: {tokens_per_sec:.1f}"
                    )

                # Evaluation
                if self.master_process and ((step > 0 and step % self.config.eval_interval == 0) or step == last_step):
                    self.model.eval()
                    val_loss = self.eval(data_loader)

                    os.makedirs(os.path.dirname(self.config.eval_log_file), exist_ok=True)
                    with open(self.config.eval_log_file, "a") as f:
                        f.write(
                            f"Step: {step * (epoch + 1)}, val_loss: {val_loss:.4f}, norm: {norm:.4f}, "
                            f"lr: {scheduler.get_last_lr()[0]:.6e}, time: {t1 - t0:.2f}s, tok/s: {tokens_per_sec:.1f}\n"
                        )

                    self.model.train()
                    if self.clean_cuda_cache and self.device.type == "cuda":
                        torch.cuda.empty_cache()

                # Save Checkpoints
                if self.master_process and ((step % self.config.checkpoints_frequency == 0 and step > 0) or step == last_step):
                    self.save_checkpoints(optimizer, self.config.path_to_checkpoints, name=str((epoch + 1) * step))

    def eval(self, data_loader):
        with torch.no_grad():
            val_loss_accum = 0.0
            k = 1

            for _ in range(self.steps_for_eval):
                x = data_loader.next_batch(split="val").to(self.device)
                x_masked, targets, mask, t = self._sample_mask(x)
                if k == 1:
                    k = 0
                    prompt_tokens = x[0][:5].unsqueeze(0).to(self.device)
                    gen_len = 100

                    mask_id = self.model.config.mask_token_id
                    cont = torch.full((1, gen_len), fill_value=mask_id, dtype=prompt_tokens.dtype, device=prompt_tokens.device)

                    x_init = torch.cat([prompt_tokens, cont], dim=1)

                    mask = torch.zeros_like(x_init, dtype=torch.bool)
                    mask[:, prompt_tokens.size(1):] = True

                    steps = 60

                    with torch.no_grad():
                        completed = self.model.generate(x_init, steps=steps, temperature=1, mask=mask, sample=False, top_k=50)

                    generated_text = self.tokenizer.decode(completed[0].tolist(), skip_special_tokens=True)
                    print(f"Prompt: '{self.tokenizer.decode(prompt_tokens[0].tolist(), skip_special_tokens=True)}'")
                    print("Generated text:\n", generated_text)

                with torch.autocast(device_type=self.device.type, dtype=self.dtype):
                    _, loss = self.model(
                        x_masked=x_masked,
                        targets=targets,
                        mask=mask,
                        t=t,
                    )
                loss = loss / self.steps_for_eval
                val_loss_accum += loss.detach()
            return val_loss_accum

    def save_checkpoints(self, optimizer, path: str, name: str):
        os.makedirs(path, exist_ok=True)
        checkpoint_path = os.path.join(path, f"model.checkpoint.{name}.pt")
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        torch.save(checkpoint, checkpoint_path)
        print("Checkpoints saved")
