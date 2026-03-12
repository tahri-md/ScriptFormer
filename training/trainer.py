import time
import math
import json
import torch
import torch.nn as nn
from pathlib import Path


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        tokenizer,
        config: dict,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer
        self.config = config

        train_cfg = config["training"]
        self.epochs = train_cfg["epochs"]
        self.max_grad_norm = train_cfg["max_grad_norm"]
        self.device = config["project"]["device"]

        self.model = self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=train_cfg["learning_rate"],
            betas=(0.9, 0.999),
            weight_decay=train_cfg["weight_decay"],
        )

        self.steps_per_epoch = len(train_loader)
        self.total_steps = self.epochs * self.steps_per_epoch
        self.warmup_steps = train_cfg["warmup_steps"]

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=self._lr_lambda,
        )

        self.checkpoint_dir = Path(train_cfg["checkpoint_dir"])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_every = train_cfg["save_every"]

        es_cfg = train_cfg["early_stopping"]
        self.early_stopping_enabled = es_cfg["enabled"]
        self.patience = es_cfg["patience"]
        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0

        self.log_every = config["logging"]["log_every"]
        self.log_dir = Path(config["logging"]["log_dir"])
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.history = {
            "train_loss": [],
            "val_loss": [],
            "learning_rates": [],
        }

        self.global_step = 0

    def _lr_lambda(self, step: int) -> float:
        if step < self.warmup_steps:
            return step / max(1, self.warmup_steps)
        else:
            progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

    def train(self):
        print(f"  ScriptFormer Training")
        print(f"  Epochs: {self.epochs} | Batch size: {self.config['training']['batch_size']}")
        print(f"  Steps/epoch: {self.steps_per_epoch} | Total steps: {self.total_steps}")
        print(f"  Warmup: {self.warmup_steps} steps | Device: {self.device}")
        print(f"  Parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(1, self.epochs + 1):
            epoch_start = time.time()

            train_loss = self._train_one_epoch(epoch)
            val_loss = self._validate(epoch)

            current_lr = self.optimizer.param_groups[0]["lr"]
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["learning_rates"].append(current_lr)

            epoch_time = time.time() - epoch_start

            print(
                f"Epoch {epoch:3d}/{self.epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"LR: {current_lr:.2e} | "
                f"Time: {epoch_time:.1f}s"
            )

            if epoch % self.save_every == 0:
                self._save_checkpoint(epoch, val_loss, is_best=False)

            if val_loss < self.best_val_loss:
                improvement = self.best_val_loss - val_loss
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                self._save_checkpoint(epoch, val_loss, is_best=True)
                print(f"New best val_loss! (improved by {improvement:.4f})")
            else:
                self.epochs_without_improvement += 1
                if self.early_stopping_enabled:
                    print(
                        f"No improvement for {self.epochs_without_improvement}"
                        f"/{self.patience} epochs"
                    )

            if (
                self.early_stopping_enabled
                and self.epochs_without_improvement >= self.patience
            ):
                print(f"\nEarly stopping triggered after {epoch} epochs!")
                print(f"Best val_loss: {self.best_val_loss:.4f}")
                break

        print("  Training Complete!")
        print(f"  Best val_loss: {self.best_val_loss:.4f}")
        print(f"  Checkpoint: {self.checkpoint_dir / 'best_model.pt'}")

        self._save_history()

        return self.history

    def _train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            images = batch["images"].to(self.device)
            token_ids = batch["token_ids"].to(self.device)

            decoder_input = token_ids[:, :-1]
            labels = token_ids[:, 1:]

            logits = self.model(images, decoder_input)

            loss = self.criterion(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
            )

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            if self.global_step % self.log_every == 0:
                avg_loss = total_loss / num_batches
                lr = self.optimizer.param_groups[0]["lr"]
                print(
                    f"  [Step {self.global_step:5d}] "
                    f"Epoch {epoch} batch {batch_idx + 1}/{self.steps_per_epoch} | "
                    f"Loss: {loss.item():.4f} (avg: {avg_loss:.4f}) | "
                    f"LR: {lr:.2e}"
                )

        return total_loss / max(1, num_batches)

    @torch.no_grad()
    def _validate(self, epoch: int) -> float:
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            images = batch["images"].to(self.device)
            token_ids = batch["token_ids"].to(self.device)

            decoder_input = token_ids[:, :-1]
            labels = token_ids[:, 1:]

            logits = self.model(images, decoder_input)

            loss = self.criterion(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
            )

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(1, num_batches)

    def _save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "val_loss": val_loss,
            "best_val_loss": self.best_val_loss,
            "global_step": self.global_step,
            "config": self.config,
        }

        if is_best:
            path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, path)

        if not is_best:
            path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
            torch.save(checkpoint, path)

    def load_checkpoint(self, checkpoint_path: str):
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint["best_val_loss"]

        epoch = checkpoint["epoch"]
        print(f"  Resumed from epoch {epoch}, best_val_loss={self.best_val_loss:.4f}")
        return epoch

    def _save_history(self):
        history_path = self.log_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"  Training history saved to {history_path}")