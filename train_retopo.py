"""
Treino em duas fases para o modelo de retopologia com K vértices fixos.
Fase 1: Reconstrução (sem Top-K) para aprender geometria.
Fase 2: Top-K ativado + losses estruturais para reduzir a exatamente TARGET_VERTICES.
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from config import (
    TRAIN_PAIRS,
    VAL_PAIRS,
    CHECKPOINTS_DIR,
    ORGANIZED_PAIRS_DIR,
    MODEL_CONFIG,
    TARGET_VERTICES,
    RETOPO_TRAINING,
    RETOPO_LOSS_WEIGHTS,
)
from dataset import get_data_loaders
from model_retopo import RetopologyGNN2000
from loss_retopo import RetopologyStructuralLoss


def train_epoch(model, loader, criterion, optimizer, device, use_topk, target_k, epoch=None):
    """
    target_k: usado como valor fixo se epoch=None (Fase 1 ou validação fixa).
    Se use_topk e epoch não for None: target por amostra = max(target_k, num_vertices_highpoly - epoch * 2).
    """
    model.train()
    total_loss = 0.0
    loss_details = {
        "chamfer_loss": 0.0,
        "hausdorff_loss": 0.0,
        "edge_length_loss": 0.0,
        "laplacian_loss": 0.0,
        "normal_consistency_loss": 0.0,
        "total_loss": 0.0,
    }
    n_batches = 0

    for batch in tqdm(loader, desc="Train"):
        highpoly = batch["highpoly"].to(device)
        lowpoly = batch["lowpoly"].to(device)

        if use_topk and epoch is not None:
            # Target dinâmico: num_vertices_highpoly - epoch * 2 (mínimo target_k)
            k = max(target_k, highpoly.num_nodes - epoch * 2)
        else:
            k = target_k
        k = min(k, highpoly.num_nodes)
        if k < 1:
            k = 1

        optimizer.zero_grad()
        pred_pos, pred_feat, importance, sel_idx, pred_edge = model(
            highpoly, use_topk=use_topk, target_vertices=k if use_topk else None
        )

        target_pos = lowpoly.pos
        target_feat = lowpoly.x[:, 3:9] if lowpoly.x.shape[1] >= 9 else None

        loss, details = criterion(
            pred_positions=pred_pos,
            pred_features=pred_feat,
            target_positions=target_pos,
            target_features=target_feat,
            pred_edge_index=pred_edge,
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        for key in loss_details:
            loss_details[key] += details.get(key, 0.0)
        n_batches += 1

    for key in loss_details:
        loss_details[key] /= max(n_batches, 1)
    return total_loss / max(n_batches, 1), loss_details


def validate(model, loader, criterion, device, use_topk, target_k, epoch=None):
    """
    Se use_topk e epoch não for None: target por amostra = max(target_k, num_vertices_highpoly - epoch * 2).
    """
    model.eval()
    total_loss = 0.0
    loss_details = {
        "chamfer_loss": 0.0,
        "hausdorff_loss": 0.0,
        "edge_length_loss": 0.0,
        "laplacian_loss": 0.0,
        "normal_consistency_loss": 0.0,
        "total_loss": 0.0,
    }
    n_batches = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Val"):
            highpoly = batch["highpoly"].to(device)
            lowpoly = batch["lowpoly"].to(device)
            if use_topk and epoch is not None:
                k = max(target_k, highpoly.num_nodes - epoch * 2)
            else:
                k = target_k
            k = min(k, highpoly.num_nodes)
            if k < 1:
                k = 1

            pred_pos, pred_feat, _, _, pred_edge = model(
                highpoly, use_topk=use_topk, target_vertices=k if use_topk else None
            )
            target_pos = lowpoly.pos
            target_feat = lowpoly.x[:, 3:9] if lowpoly.x.shape[1] >= 9 else None

            loss, details = criterion(
                pred_positions=pred_pos,
                pred_features=pred_feat,
                target_positions=target_pos,
                target_features=target_feat,
                pred_edge_index=pred_edge,
            )

            total_loss += loss.item()
            for key in loss_details:
                loss_details[key] += details.get(key, 0.0)
            n_batches += 1

    for key in loss_details:
        loss_details[key] /= max(n_batches, 1)
    return total_loss / max(n_batches, 1), loss_details


def save_checkpoint(model, optimizer, scheduler, epoch, phase, loss, checkpoint_dir, is_best=False, history=None, best_val_loss=None):
    ckpt = {
        "epoch": epoch,
        "phase": phase,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "model_config": MODEL_CONFIG,
        "target_vertices": TARGET_VERTICES,
    }
    if best_val_loss is not None:
        ckpt["best_val_loss"] = best_val_loss
    if scheduler is not None:
        ckpt["scheduler_state_dict"] = scheduler.state_dict()
    if history is not None:
        ckpt["history"] = history

    path = checkpoint_dir / f"retopo_epoch_{epoch}.pt"
    torch.save(ckpt, path)
    last_path = checkpoint_dir / "retopo_last.pt"
    torch.save(ckpt, last_path)
    if is_best:
        best_path = checkpoint_dir / "retopo_best.pt"
        torch.save(ckpt, best_path)
        print(f"✓ Melhor modelo retopo salvo em {best_path}")
    return path


def main():
    parser = argparse.ArgumentParser(description="Treino em duas fases: modelo retopologia K vértices")
    parser.add_argument("--resume", "-r", type=str, default=None, help="Checkpoint retopo para continuar")
    parser.add_argument("--phase1-only", action="store_true", help="Correr apenas Fase 1")
    parser.add_argument("--phase2-only", action="store_true", help="Correr apenas Fase 2 (exige checkpoint Fase 1)")
    parser.add_argument("--output-dir", "-o", type=str, default=None, help="Diretório de checkpoints (default: checkpoints/retopo)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir) if args.output_dir else CHECKPOINTS_DIR / "retopo"
    out_dir.mkdir(parents=True, exist_ok=True)

    phase1_epochs = RETOPO_TRAINING["phase1_epochs"]
    phase2_epochs = RETOPO_TRAINING["phase2_epochs"]
    bs1 = RETOPO_TRAINING["batch_size_phase1"]
    bs2 = RETOPO_TRAINING["batch_size_phase2"]
    lr = RETOPO_TRAINING["learning_rate"]

    all_pairs = [f"{i:04d}" for i in range(1, TRAIN_PAIRS + VAL_PAIRS + 1)]
    train_pairs = all_pairs[:TRAIN_PAIRS]
    val_pairs = all_pairs[TRAIN_PAIRS : TRAIN_PAIRS + VAL_PAIRS]

    train_loader_1, val_loader_1 = get_data_loaders(train_pairs, val_pairs, batch_size=bs1)
    train_loader_2, val_loader_2 = get_data_loaders(train_pairs, val_pairs, batch_size=bs2)

    model = RetopologyGNN2000(MODEL_CONFIG, target_vertices=TARGET_VERTICES).to(device)
    criterion = RetopologyStructuralLoss(**RETOPO_LOSS_WEIGHTS)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

    history = {"train_loss": [], "val_loss": [], "epochs": [], "phase": []}
    start_epoch = 1
    best_val = float("inf")
    run_phase1 = not args.phase2_only
    run_phase2 = not args.phase1_only

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt and scheduler is not None:
            try:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            except Exception:
                pass
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val = ckpt.get("best_val_loss", best_val)
        history = ckpt.get("history", history)
        print(f"✓ Checkpoint carregado; a continuar da época {start_epoch}")

    print("=" * 70)
    print("Treino Retopologia (2 fases) — K =", TARGET_VERTICES)
    print("=" * 70)

    if args.phase2_only:
        start_epoch = 1

    t0 = time.time()

    # ----- Fase 1 -----
    if run_phase1:
        print("\n--- Fase 1: Reconstrução (sem Top-K) ---")
        for epoch in range(start_epoch, phase1_epochs + 1):
            train_loss, train_d = train_epoch(
                model, train_loader_1, criterion, optimizer, device,
                use_topk=False, target_k=TARGET_VERTICES, epoch=None,
            )
            history["train_loss"].append(train_loss)
            history["phase"].append(1)

            val_loss, val_d = validate(
                model, val_loader_1, criterion, device,
                use_topk=False, target_k=TARGET_VERTICES, epoch=None,
            )
            history["val_loss"].append(val_loss)
            history["epochs"].append(epoch)

            scheduler.step(val_loss)
            if val_loss < best_val:
                best_val = val_loss
            save_checkpoint(model, optimizer, scheduler, epoch, 1, train_loss, out_dir, is_best=(val_loss == best_val), history=history, best_val_loss=best_val)

            print(f"Epoch {epoch} [Fase 1] train={train_loss:.6f} val={val_loss:.6f} chamfer={train_d['chamfer_loss']:.6f}")

        start_epoch = phase1_epochs + 1
        if args.phase1_only:
            print("Fase 1 concluída (--phase1-only). Para Fase 2: python train_retopo.py --phase2-only -r checkpoints/retopo/retopo_last.pt")
            return

    # ----- Fase 2 -----
    if run_phase2:
        phase2_start = (phase1_epochs + 1) if run_phase1 else 1
        phase2_end = phase2_start + phase2_epochs
        epoch_start = max(phase2_start, start_epoch)
        print("\n--- Fase 2: Top-K ativo (exatamente", TARGET_VERTICES, "vértices) ---")
        for epoch in range(epoch_start, phase2_end):
            # Target dinâmico por época: max(TARGET_VERTICES, num_vertices_highpoly - epoch * 2)
            train_loss, train_d = train_epoch(
                model, train_loader_2, criterion, optimizer, device,
                use_topk=True, target_k=TARGET_VERTICES, epoch=epoch,
            )
            history["train_loss"].append(train_loss)
            history["phase"].append(2)

            val_loss, val_d = validate(
                model, val_loader_2, criterion, device,
                use_topk=True, target_k=TARGET_VERTICES, epoch=epoch,
            )
            history["val_loss"].append(val_loss)
            history["epochs"].append(epoch)

            scheduler.step(val_loss)
            if val_loss < best_val:
                best_val = val_loss
            save_checkpoint(model, optimizer, scheduler, epoch, 2, train_loss, out_dir, is_best=(val_loss == best_val), history=history, best_val_loss=best_val)

            print(f"Epoch {epoch} [Fase 2] train={train_loss:.6f} val={val_loss:.6f} chamfer={train_d['chamfer_loss']:.6f} edge={train_d['edge_length_loss']:.6f}")

    with open(out_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print("=" * 70)
    print(f"Treino concluído em {(time.time() - t0) / 60:.2f} min. Melhor val loss: {best_val:.6f}")
    print("Checkpoints em:", out_dir)
    print("=" * 70)


if __name__ == "__main__":
    main()
