"""
Treino do ModelRetopology2000: highpoly + heatmap (importância) -> lowpoly.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import time
import argparse

from config import (
    TRAIN_PAIRS, VAL_PAIRS, TRAINING_CONFIG, MODEL_CONFIG,
    CHECKPOINTS_DIR, ORGANIZED_PAIRS_DIR,
)
from dataset import get_data_loaders
from model import RetopologyGNN2000


class RetopologyLoss(nn.Module):
    """Chamfer para posições + MSE para features (normais)."""

    def __init__(self, pos_weight=2, normal_weight=0.1):
        super().__init__()
        self.pos_weight = pos_weight
        self.normal_weight = normal_weight
        self.mse = nn.MSELoss()

    def forward(self, pred_positions, pred_features, target_positions, target_features=None, **kwargs):
        pos_loss = self._chamfer(pred_positions, target_positions)
        if target_features is not None and pred_features.shape[0] == target_features.shape[0]:
            normal_loss = self.mse(pred_features[:, :3], target_features[:, :3])
            feat_rest = self.mse(pred_features[:, 3:], target_features[:, 3:6])
            feature_total = normal_loss + feat_rest
        else:
            feature_total = torch.tensor(0.0, device=pred_positions.device)
        total = self.pos_weight * pos_loss + self.normal_weight * feature_total
        return total, {
            "pos_loss": pos_loss.item(),
            "feature_loss": feature_total.item() if isinstance(feature_total, torch.Tensor) else 0.0,
            "total_loss": total.item(),
        }

    def _chamfer(self, pred, target):
        d = torch.cdist(pred, target)
        return torch.mean(d.min(dim=1)[0]) + torch.mean(d.min(dim=0)[0])


def train_epoch(model, train_loader, criterion, optimizer, device, grad_clip_max_norm=1.0):
    model.train()
    total_loss = 0.0
    loss_details = {"pos_loss": 0.0, "feature_loss": 0.0, "total_loss": 0.0}
    num_batches = 0
    for batch in tqdm(train_loader, desc="Treino"):
        highpoly = batch["highpoly"].to(device)
        lowpoly = batch["lowpoly"].to(device)
        optimizer.zero_grad()
        pred_positions, pred_features = model(highpoly)
        target_positions = lowpoly.pos
        target_features = lowpoly.x[:, 3:9] if lowpoly.x.shape[1] >= 9 else None
        loss, loss_dict = criterion(
            pred_positions, pred_features,
            target_positions, target_features,
        )
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
    loss_details = {"pos_loss": 0.0, "feature_loss": 0.0, "total_loss": 0.0}
    num_batches = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validação"):
            highpoly = batch["highpoly"].to(device)
            lowpoly = batch["lowpoly"].to(device)
            pred_positions, pred_features = model(highpoly)
            target_positions = lowpoly.pos
            target_features = lowpoly.x[:, 3:9] if lowpoly.x.shape[1] >= 9 else None
            loss, loss_dict = criterion(
                pred_positions, pred_features,
                target_positions, target_features,
            )
            total_loss += loss.item()
            for k in loss_details:
                loss_details[k] += loss_dict.get(k, 0.0)
            num_batches += 1
    n = num_batches or 1
    return total_loss / n, {k: loss_details[k] / n for k in loss_details}


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, device="cpu"):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    epoch = ckpt.get("epoch", 0)
    best_val_loss = ckpt.get("best_val_loss", float("inf"))
    history = ckpt.get("history", {"train_loss": [], "val_loss": [], "epochs": []})
    return epoch, best_val_loss, history


def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir, is_best=False, scheduler=None, best_val_loss=None, history=None):
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "model_config": MODEL_CONFIG,
        "training_config": TRAINING_CONFIG,
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
    parser = argparse.ArgumentParser(description="Treinar ModelRetopology2000 (heatmap + highpoly -> lowpoly)")
    parser.add_argument("--resume", "-r", type=str, default=None, help="Checkpoint para continuar")
    args = parser.parse_args()

    print("=" * 70)
    print("ModelRetopology2000: Treino (heatmap + highpoly -> lowpoly)")
    print("=" * 70)

    device = torch.device(TRAINING_CONFIG["device"] if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")
    print(f"Pares treino: {TRAIN_PAIRS}, validação: {VAL_PAIRS}")
    print(f"Pasta pares: {ORGANIZED_PAIRS_DIR}")
    print()

    all_pairs = [f"{i:04d}" for i in range(1, TRAIN_PAIRS + VAL_PAIRS + 1)]
    train_pairs = all_pairs[:TRAIN_PAIRS]
    val_pairs = all_pairs[TRAIN_PAIRS : TRAIN_PAIRS + VAL_PAIRS]

    train_loader, val_loader = get_data_loaders(
        train_pairs, val_pairs, batch_size=TRAINING_CONFIG["batch_size"]
    )
    model = RetopologyGNN2000(MODEL_CONFIG).to(device)
    criterion = RetopologyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=TRAINING_CONFIG["learning_rate"],
        weight_decay=TRAINING_CONFIG.get("weight_decay", 0),
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
    history = {"train_loss": [], "val_loss": [], "epochs": []}
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

    for epoch in range(start_epoch, TRAINING_CONFIG["num_epochs"] + 1):
        print(f"\nÉpoca {epoch}/{TRAINING_CONFIG['num_epochs']}")
        train_loss, train_details = train_epoch(
            model, train_loader, criterion, optimizer, device, grad_clip_max_norm=grad_clip
        )
        history["train_loss"].append(train_loss)
        print(f"Train loss: {train_loss:.6f} (pos: {train_details['pos_loss']:.6f}, feat: {train_details['feature_loss']:.6f})")

        if epoch % TRAINING_CONFIG["val_every"] == 0:
            val_loss, val_details = validate(model, val_loader, criterion, device)
            history["val_loss"].append(val_loss)
            history["epochs"].append(epoch)
            print(f"Val loss: {val_loss:.6f}")
            scheduler.step(val_loss)
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            save_checkpoint(
                model, optimizer, epoch, train_loss, CHECKPOINTS_DIR,
                is_best=is_best, scheduler=scheduler, best_val_loss=best_val_loss, history=history,
            )
            if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
                print(f"\n⏹ Early stopping: sem melhoria em {early_stopping_patience} validações. A parar.")
                break
        else:
            if epoch % TRAINING_CONFIG["save_every"] == 0:
                save_checkpoint(
                    model, optimizer, epoch, train_loss, CHECKPOINTS_DIR,
                    is_best=False, scheduler=scheduler, best_val_loss=best_val_loss, history=history,
                )

        with open(CHECKPOINTS_DIR / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)

    save_checkpoint(
        model, optimizer, TRAINING_CONFIG["num_epochs"], train_loss, CHECKPOINTS_DIR,
        is_best=False, scheduler=scheduler, best_val_loss=best_val_loss, history=history,
    )
    print(f"\nTreino concluído. Tempo: {(time.time() - start_time) / 60:.2f} min. Melhor val loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    main()
