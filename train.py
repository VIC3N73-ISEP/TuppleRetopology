"""
Script de treinamento para o modelo GNN de retopologia.
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

from config import (
    TRAIN_PAIRS, VAL_PAIRS, TRAINING_CONFIG, MODEL_CONFIG,
    CHECKPOINTS_DIR, ORGANIZED_PAIRS_DIR
)
from model import RetopologyGNN, ImprovedRetopologyGNN
from dataset import get_data_loaders


class RetopologyLoss(nn.Module):
    """
    Loss function customizada para retopologia.
    Combina múltiplas métricas: posição, normal, preservação de forma.
    """
    
    def __init__(self, pos_weight=1.0, normal_weight=0.5, shape_weight=0.3):
        super(RetopologyLoss, self).__init__()
        self.pos_weight = pos_weight
        self.normal_weight = normal_weight
        self.shape_weight = shape_weight
        
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
    
    def forward(self, pred_positions, pred_features, target_positions, target_features=None):
        """
        Calcula a loss combinada.
        
        Args:
            pred_positions: Posições preditas (N, 3)
            pred_features: Features preditas (N, 6)
            target_positions: Posições alvo (M, 3)
            target_features: Features alvo (M, 6) - opcional
        """
        # Loss de posição (usar chamfer distance aproximada)
        pos_loss = self._chamfer_distance(pred_positions, target_positions)
        
        # Loss de features (normais, etc.)
        if target_features is not None and pred_features.shape[0] == target_features.shape[0]:
            # Normal loss (primeiros 3 componentes)
            normal_loss = self.mse_loss(pred_features[:, :3], target_features[:, :3])
            # Outras features
            feature_loss = self.mse_loss(pred_features[:, 3:], target_features[:, 3:])
            feature_total = normal_loss + feature_loss
        else:
            feature_total = torch.tensor(0.0, device=pred_positions.device)
        
        # Loss total
        total_loss = (
            self.pos_weight * pos_loss +
            self.normal_weight * feature_total
        )
        
        return total_loss, {
            "pos_loss": pos_loss.item(),
            "feature_loss": feature_total.item() if isinstance(feature_total, torch.Tensor) else 0.0,
            "total_loss": total_loss.item()
        }
    
    def _chamfer_distance(self, pred, target):
        """
        Calcula uma aproximação da Chamfer Distance.
        """
        # Para cada ponto em pred, encontrar o ponto mais próximo em target
        distances_pred_to_target = torch.cdist(pred, target)
        min_dist_pred = torch.min(distances_pred_to_target, dim=1)[0]
        
        # Para cada ponto em target, encontrar o ponto mais próximo em pred
        min_dist_target = torch.min(distances_pred_to_target, dim=0)[0]
        
        # Chamfer distance
        chamfer = torch.mean(min_dist_pred) + torch.mean(min_dist_target)
        
        return chamfer


def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Treina o modelo por uma época.
    """
    model.train()
    total_loss = 0.0
    loss_details = {"pos_loss": 0.0, "feature_loss": 0.0, "total_loss": 0.0}
    num_batches = 0
    
    pbar = tqdm(train_loader, desc="Treinando")
    for batch in pbar:
        highpoly = batch["highpoly"].to(device)
        lowpoly = batch["lowpoly"].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        pred_positions, pred_features = model(highpoly)
        
        # Calcular loss (usar posições e features do lowpoly alvo)
        target_positions = lowpoly.pos
        target_features = lowpoly.x[:, 3:9] if lowpoly.x.shape[1] >= 9 else None
        
        loss, loss_dict = criterion(
            pred_positions, pred_features,
            target_positions, target_features
        )
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Acumular estatísticas
        total_loss += loss.item()
        for key in loss_details:
            loss_details[key] += loss_dict.get(key, 0.0)
        num_batches += 1
        
        # Atualizar progress bar
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "pos": f"{loss_dict['pos_loss']:.4f}"
        })
    
    # Médias
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    for key in loss_details:
        loss_details[key] /= num_batches if num_batches > 0 else 1.0
    
    return avg_loss, loss_details


def validate(model, val_loader, criterion, device):
    """
    Valida o modelo.
    """
    model.eval()
    total_loss = 0.0
    loss_details = {"pos_loss": 0.0, "feature_loss": 0.0, "total_loss": 0.0}
    num_batches = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validando")
        for batch in pbar:
            highpoly = batch["highpoly"].to(device)
            lowpoly = batch["lowpoly"].to(device)
            
            # Forward pass
            pred_positions, pred_features = model(highpoly)
            
            # Calcular loss
            target_positions = lowpoly.pos
            target_features = lowpoly.x[:, 3:9] if lowpoly.x.shape[1] >= 9 else None
            
            loss, loss_dict = criterion(
                pred_positions, pred_features,
                target_positions, target_features
            )
            
            # Acumular estatísticas
            total_loss += loss.item()
            for key in loss_details:
                loss_details[key] += loss_dict.get(key, 0.0)
            num_batches += 1
            
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "pos": f"{loss_dict['pos_loss']:.4f}"
            })
    
    # Médias
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    for key in loss_details:
        loss_details[key] /= num_batches if num_batches > 0 else 1.0
    
    return avg_loss, loss_details


def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir, is_best=False):
    """
    Salva um checkpoint do modelo.
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "model_config": MODEL_CONFIG,
        "training_config": TRAINING_CONFIG
    }
    
    # Salvar checkpoint regular
    checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
    torch.save(checkpoint, checkpoint_path)
    
    # Salvar melhor modelo
    if is_best:
        best_path = checkpoint_dir / "best_model.pt"
        torch.save(checkpoint, best_path)
        print(f"✓ Melhor modelo salvo em {best_path}")
    
    # Salvar último checkpoint
    last_path = checkpoint_dir / "last_checkpoint.pt"
    torch.save(checkpoint, last_path)
    
    return checkpoint_path


def main():
    """
    Função principal de treinamento.
    """
    print("=" * 70)
    print("Treinamento do Modelo GNN para Retopologia")
    print("=" * 70)
    print()
    
    # Configurações
    device = torch.device(TRAINING_CONFIG["device"] if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")
    print(f"Pares para treino: {TRAIN_PAIRS}")
    print(f"Pares para validação: {VAL_PAIRS}")
    print()
    
    # Gerar listas de pares
    all_pairs = [f"{i:04d}" for i in range(1, TRAIN_PAIRS + VAL_PAIRS + 1)]
    train_pairs = all_pairs[:TRAIN_PAIRS]
    val_pairs = all_pairs[TRAIN_PAIRS:TRAIN_PAIRS + VAL_PAIRS]
    
    print(f"Pares de treino: {train_pairs[:5]}... (total: {len(train_pairs)})")
    print(f"Pares de validação: {val_pairs}")
    print()
    
    # Criar data loaders
    print("Carregando datasets...")
    train_loader, val_loader = get_data_loaders(
        train_pairs, val_pairs, batch_size=TRAINING_CONFIG["batch_size"]
    )
    print(f"✓ Datasets carregados")
    print()
    
    # Criar modelo
    print("Inicializando modelo...")
    model = RetopologyGNN(MODEL_CONFIG).to(device)
    print(f"✓ Modelo criado com {sum(p.numel() for p in model.parameters())} parâmetros")
    print()
    
    # Loss e otimizador
    criterion = RetopologyLoss()
    optimizer = optim.Adam(model.parameters(), lr=TRAINING_CONFIG["learning_rate"])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # Histórico de treinamento
    history = {
        "train_loss": [],
        "val_loss": [],
        "epochs": []
    }
    
    best_val_loss = float('inf')
    start_time = time.time()
    
    # Loop de treinamento
    print("Iniciando treinamento...")
    print("=" * 70)
    
    for epoch in range(1, TRAINING_CONFIG["num_epochs"] + 1):
        print(f"\nÉpoca {epoch}/{TRAINING_CONFIG['num_epochs']}")
        print("-" * 70)
        
        # Treinar
        train_loss, train_details = train_epoch(model, train_loader, criterion, optimizer, device)
        history["train_loss"].append(train_loss)
        
        print(f"Loss de treino: {train_loss:.6f}")
        print(f"  - Posição: {train_details['pos_loss']:.6f}")
        print(f"  - Features: {train_details['feature_loss']:.6f}")
        
        # Validar
        if epoch % TRAINING_CONFIG["val_every"] == 0:
            val_loss, val_details = validate(model, val_loader, criterion, device)
            history["val_loss"].append(val_loss)
            history["epochs"].append(epoch)
            
            print(f"Loss de validação: {val_loss:.6f}")
            print(f"  - Posição: {val_details['pos_loss']:.6f}")
            print(f"  - Features: {val_details['feature_loss']:.6f}")
            
            # Atualizar learning rate
            scheduler.step(val_loss)
            
            # Salvar melhor modelo
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                print(f"✓ Novo melhor modelo! (loss: {best_val_loss:.6f})")
        else:
            val_loss = None
        
        # Salvar checkpoint
        if epoch % TRAINING_CONFIG["save_every"] == 0:
            save_checkpoint(model, optimizer, epoch, train_loss, CHECKPOINTS_DIR, is_best=False)
            print(f"✓ Checkpoint salvo (época {epoch})")
        
        # Salvar histórico
        history_path = CHECKPOINTS_DIR / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
    
    # Salvar checkpoint final
    save_checkpoint(model, optimizer, TRAINING_CONFIG["num_epochs"], train_loss, CHECKPOINTS_DIR, is_best=False)
    
    elapsed_time = time.time() - start_time
    print()
    print("=" * 70)
    print("Treinamento concluído!")
    print(f"Tempo total: {elapsed_time/60:.2f} minutos")
    print(f"Melhor loss de validação: {best_val_loss:.6f}")
    print(f"Checkpoints salvos em: {CHECKPOINTS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()

