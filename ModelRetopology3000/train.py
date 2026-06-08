"""
Treino ModelRetopology3000 (decimate-only): aprende apenas a prever ratio.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
import json
from tqdm import tqdm
import time
import argparse

from config import (
    TRAIN_PAIRS,
    VAL_PAIRS,
    TRAINING_CONFIG,
    MODEL_CONFIG,
    LOSS_CONFIG,
    CHECKPOINTS_DIR,
    ORGANIZED_PAIRS_DIR,
)
from dataset import get_data_loaders
from model import RetopologyModel3000


class RatioLoss(nn.Module):
    def __init__(self, w_ratio=1.0):
        super().__init__()
        self.w_ratio = w_ratio
        self.mse = nn.MSELoss()

    def forward(self, r_hat, r_gt, r_applied_gt):
        # Loss explícita na semântica da inferência:
        # ratio_applied = 1 - ratio_final
        r_applied_hat = 1.0 - r_hat
        ratio_applied_loss = self.mse(r_applied_hat, r_applied_gt)
        # Métrica equivalente no espaço final (útil para debug/monitoring)
        ratio_final_loss = self.mse(r_hat, r_gt)
        total = self.w_ratio * ratio_applied_loss
        return total, {
            "ratio_final_loss": ratio_final_loss.item(),
            "ratio_applied_loss": ratio_applied_loss.item(),
            "r_final_mean": r_hat.mean().item(),
            "r_applied_mean": r_applied_hat.mean().item(),
            "total_loss": total.item(),
        }


def train_epoch(model, train_loader, criterion, optimizer, device, grad_clip_max_norm=1.0):
    model.train()
    total_loss = 0.0
    loss_details = {
        "ratio_final_loss": 0.0,
        "ratio_applied_loss": 0.0,
        "r_final_mean": 0.0,
        "r_applied_mean": 0.0,
        "total_loss": 0.0,
    }
    num_batches = 0
    for batch in tqdm(train_loader, desc="Treino"):
        high = batch["high"].to(device)
        r_gt = batch["r_gt"].to(device)
        r_applied_gt = batch["r_applied_gt"].to(device)
        optimizer.zero_grad()
        r_hat = model(high)
        loss, loss_dict = criterion(r_hat, r_gt, r_applied_gt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_max_norm)
        optimizer.step()
        total_loss += loss.item()
        for k in loss_details:
            loss_details[k] += loss_dict.get(k, 0.0)
        num_batches += 1
    n = num_batches or 1
    return total_loss / n, {k: loss_details[k] / n for k in loss_details}


def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    loss_details = {
        "ratio_final_loss": 0.0,
        "ratio_applied_loss": 0.0,
        "r_final_mean": 0.0,
        "r_applied_mean": 0.0,
        "total_loss": 0.0,
    }
    num_batches = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validação"):
            high = batch["high"].to(device)
            r_gt = batch["r_gt"].to(device)
            r_applied_gt = batch["r_applied_gt"].to(device)
            r_hat = model(high)
            loss, loss_dict = criterion(r_hat, r_gt, r_applied_gt)
            total_loss += loss.item()
            for k in loss_details:
                loss_details[k] += loss_dict.get(k, 0.0)
            num_batches += 1
    n = num_batches or 1
    return total_loss / n, {k: loss_details[k] / n for k in loss_details}


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, device="cpu"):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    epoch = ckpt.get("epoch", 0)
    best_val_loss = ckpt.get("best_val_loss", float("inf"))
    history = ckpt.get(
        "history",
        {
            "train_loss": [],
            "val_loss": [],
            "epochs": [],
            "train_ratio_final_loss": [],
            "train_ratio_applied_loss": [],
            "train_r_final_mean": [],
            "train_r_applied_mean": [],
            "val_ratio_final_loss": [],
            "val_ratio_applied_loss": [],
            "val_r_final_mean": [],
            "val_r_applied_mean": [],
        },
    )
    return epoch, best_val_loss, history


def save_checkpoint(
    model,
    optimizer,
    epoch,
    loss,
    checkpoint_dir,
    is_best=False,
    scheduler=None,
    best_val_loss=None,
    history=None,
):
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "model_config": MODEL_CONFIG,
        "training_config": TRAINING_CONFIG,
        "loss_config": LOSS_CONFIG,
    }
    if scheduler is not None:
        ckpt["scheduler_state_dict"] = scheduler.state_dict()
    if best_val_loss is not None:
        ckpt["best_val_loss"] = best_val_loss
    if history is not None:
        ckpt["history"] = history
    path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
    torch.save(ckpt, path)
    if is_best:
        torch.save(ckpt, checkpoint_dir / "best_model.pt")
        print(f"✓ Melhor modelo salvo em {checkpoint_dir / 'best_model.pt'}")
    torch.save(ckpt, checkpoint_dir / "last_checkpoint.pt")
    return path


def main():
    parser = argparse.ArgumentParser(description="Treinar ModelRetopology3000 (decimate-only ratio predictor)")
    parser.add_argument("--resume", "-r", type=str, default=None, help="Checkpoint para continuar")
    args = parser.parse_args()

    print("=" * 70)
    print("ModelRetopology3000: Treino ratio-only (decimate-only)")
    print("=" * 70)
    print("Semântica: ratio_applied = 1 - ratio_final")

    device = torch.device(TRAINING_CONFIG["device"] if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")
    print(f"Pares treino: {TRAIN_PAIRS}, validação: {VAL_PAIRS}")
    print(f"Pasta pares: {ORGANIZED_PAIRS_DIR}")
    print(f"LOSS_CONFIG: {LOSS_CONFIG}")
    print()

    all_pairs = [f"{i:04d}" for i in range(1, TRAIN_PAIRS + VAL_PAIRS + 1)]
    train_pairs = all_pairs[:TRAIN_PAIRS]
    val_pairs = all_pairs[TRAIN_PAIRS : TRAIN_PAIRS + VAL_PAIRS]

    train_loader, val_loader = get_data_loaders(
        train_pairs, val_pairs, batch_size=TRAINING_CONFIG["batch_size"]
    )
    model = RetopologyModel3000(MODEL_CONFIG).to(device)
    criterion = RatioLoss(w_ratio=LOSS_CONFIG.get("w_ratio", 1.0))
    optimizer = optim.Adam(
        model.parameters(),
        lr=TRAINING_CONFIG["learning_rate"],
        weight_decay=TRAINING_CONFIG.get("weight_decay", 0),
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
    history = {
        "train_loss": [],
        "val_loss": [],
        "epochs": [],
        "train_ratio_final_loss": [],
        "train_ratio_applied_loss": [],
        "train_r_final_mean": [],
        "train_r_applied_mean": [],
        "val_ratio_final_loss": [],
        "val_ratio_applied_loss": [],
        "val_r_final_mean": [],
        "val_r_applied_mean": [],
    }
    best_val_loss = float("inf")
    start_epoch = 1
    start_time = time.time()
    epochs_without_improvement = 0
    early_stopping_patience = TRAINING_CONFIG.get("early_stopping_patience", 0)
    grad_clip = TRAINING_CONFIG.get("grad_clip_max_norm", 1.0)

    if args.resume:
        rpath = Path(args.resume)
        if not rpath.exists():
            print(f"Checkpoint não encontrado: {rpath}")
            return
        start_epoch, best_val_loss, history = load_checkpoint(rpath, model, optimizer, scheduler, device)
        start_epoch += 1
        print(f"Continuando da época {start_epoch}")

    train_loss = 0.0
    for epoch in range(start_epoch, TRAINING_CONFIG["num_epochs"] + 1):
        print(f"\nÉpoca {epoch}/{TRAINING_CONFIG['num_epochs']}")
        train_loss, train_details = train_epoch(
            model, train_loader, criterion, optimizer, device, grad_clip_max_norm=grad_clip
        )
        history["train_loss"].append(train_loss)
        history["train_ratio_final_loss"].append(train_details["ratio_final_loss"])
        history["train_ratio_applied_loss"].append(train_details["ratio_applied_loss"])
        history["train_r_final_mean"].append(train_details["r_final_mean"])
        history["train_r_applied_mean"].append(train_details["r_applied_mean"])
        print(
            f"Train loss: {train_loss:.6f} "
            f"(final_loss: {train_details['ratio_final_loss']:.6f}, "
            f"applied_loss: {train_details['ratio_applied_loss']:.6f}, "
            f"r_final_mean: {train_details['r_final_mean']:.4f}, "
            f"r_apply_mean: {train_details['r_applied_mean']:.4f})"
        )

        if epoch % TRAINING_CONFIG["val_every"] == 0:
            val_loss, val_details = validate(model, val_loader, criterion, device)
            history["val_loss"].append(val_loss)
            history["epochs"].append(epoch)
            history["val_ratio_final_loss"].append(val_details["ratio_final_loss"])
            history["val_ratio_applied_loss"].append(val_details["ratio_applied_loss"])
            history["val_r_final_mean"].append(val_details["r_final_mean"])
            history["val_r_applied_mean"].append(val_details["r_applied_mean"])
            print(
                f"Val loss: {val_loss:.6f} "
                f"(final_loss: {val_details['ratio_final_loss']:.6f}, "
                f"applied_loss: {val_details['ratio_applied_loss']:.6f}, "
                f"r_final_mean: {val_details['r_final_mean']:.4f}, "
                f"r_apply_mean: {val_details['r_applied_mean']:.4f})"
            )
            scheduler.step(val_loss)
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            save_checkpoint(
                model,
                optimizer,
                epoch,
                train_loss,
                CHECKPOINTS_DIR,
                is_best=is_best,
                scheduler=scheduler,
                best_val_loss=best_val_loss,
                history=history,
            )
            if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
                print(f"\n⏹ Early stopping: sem melhoria em {early_stopping_patience} validações. A parar.")
                break
        else:
            if epoch % TRAINING_CONFIG["save_every"] == 0:
                save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    train_loss,
                    CHECKPOINTS_DIR,
                    is_best=False,
                    scheduler=scheduler,
                    best_val_loss=best_val_loss,
                    history=history,
                )

        with open(CHECKPOINTS_DIR / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)

    save_checkpoint(
        model,
        optimizer,
        TRAINING_CONFIG["num_epochs"],
        train_loss,
        CHECKPOINTS_DIR,
        is_best=False,
        scheduler=scheduler,
        best_val_loss=best_val_loss,
        history=history,
    )
    print(
        f"\nTreino concluído. Tempo: {(time.time() - start_time) / 60:.2f} min. "
        f"Melhor val loss: {best_val_loss:.6f}"
    )


if __name__ == "__main__":
    main()
