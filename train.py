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
import argparse

from config import (
    TRAIN_PAIRS, VAL_PAIRS, TRAINING_CONFIG, MODEL_CONFIG,
    CHECKPOINTS_DIR, ORGANIZED_PAIRS_DIR
)

# Importar modelo: tentar novo, fallback para antigo
try:
    from model_v2 import StableRetopologyGNN as RetopologyGNN
    USE_NEW_MODEL = True
except ImportError:
    from model import RetopologyGNN
    USE_NEW_MODEL = False

# Importar loss: tentar novo, fallback para antigo
try:
    from loss_v2 import ImprovedRetopologyLoss as RetopologyLoss
    USE_NEW_LOSS = True
except ImportError:
    from train import RetopologyLoss
    USE_NEW_LOSS = False

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
    # Inicializar loss_details com todas as chaves possíveis (compatível com loss antiga e nova)
    loss_details = {
        "pos_loss": 0.0, "chamfer_loss": 0.0, "hausdorff_loss": 0.0,
        "normal_consistency_loss": 0.0, "edge_length_loss": 0.0,
        "degenerate_loss": 0.0, "smoothness_loss": 0.0,
        "vertex_count_loss": 0.0, "feature_loss": 0.0, "total_loss": 0.0
    }
    num_batches = 0
    
    pbar = tqdm(train_loader, desc="Treinando")
    for batch in pbar:
        highpoly = batch["highpoly"].to(device)
        lowpoly = batch["lowpoly"].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        
        # Obter target vertex count do lowpoly
        target_vertex_count = lowpoly.pos.shape[0]
        
        # Forward pass (compatível com modelo antigo e novo)
        if USE_NEW_MODEL:
            pred_positions, pred_features, importance_scores, selected_indices = model(
                highpoly, target_vertex_count=target_vertex_count
            )
            # Filtrar edge_index para usar apenas vértices selecionados
            # Criar mapeamento: índice original → novo índice
            n_original = highpoly.edge_index.max().item() + 1 if highpoly.edge_index.numel() > 0 else 0
            if n_original > 0:
                mapping = torch.full((n_original,), -1, dtype=torch.long, device=highpoly.edge_index.device)
                mapping[selected_indices] = torch.arange(len(selected_indices), device=highpoly.edge_index.device)
                
                # Filtrar arestas onde ambos os vértices estão selecionados
                mask = (mapping[highpoly.edge_index[0]] >= 0) & (mapping[highpoly.edge_index[1]] >= 0)
                filtered_edges = highpoly.edge_index[:, mask]
                
                # Remapear índices
                if filtered_edges.numel() > 0:
                    filtered_edges[0] = mapping[filtered_edges[0]]
                    filtered_edges[1] = mapping[filtered_edges[1]]
                    pred_edge_index = filtered_edges
                else:
                    pred_edge_index = torch.empty((2, 0), dtype=torch.long, device=highpoly.edge_index.device)
            else:
                pred_edge_index = torch.empty((2, 0), dtype=torch.long, device=highpoly.edge_index.device)
        else:
            pred_positions, pred_features = model(highpoly)
            pred_edge_index = highpoly.edge_index
        
        # Calcular loss (usar posições e features do lowpoly alvo)
        target_positions = lowpoly.pos
        target_features = lowpoly.x[:, 3:9] if lowpoly.x.shape[1] >= 9 else None
        
        # Preparar argumentos da loss (compatível com versão antiga e nova)
        loss_args = {
            "pred_positions": pred_positions,
            "pred_features": pred_features,
            "target_positions": target_positions,
            "target_features": target_features
        }
        
        # Adicionar argumentos extras se usar nova loss
        if USE_NEW_LOSS:
            loss_args.update({
                "pred_edge_index": pred_edge_index,
                "target_edge_index": lowpoly.edge_index,
                "target_vertex_count": target_vertex_count
            })
        
        loss, loss_dict = criterion(**loss_args)
        
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
        pos_loss_key = 'chamfer_loss' if 'chamfer_loss' in loss_dict else 'pos_loss'
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "pos": f"{loss_dict.get(pos_loss_key, 0):.4f}"
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
    # Inicializar loss_details com todas as chaves possíveis (compatível com loss antiga e nova)
    loss_details = {
        "pos_loss": 0.0, "chamfer_loss": 0.0, "hausdorff_loss": 0.0,
        "normal_consistency_loss": 0.0, "edge_length_loss": 0.0,
        "degenerate_loss": 0.0, "smoothness_loss": 0.0,
        "vertex_count_loss": 0.0, "feature_loss": 0.0, "total_loss": 0.0
    }
    num_batches = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validando")
        for batch in pbar:
            highpoly = batch["highpoly"].to(device)
            lowpoly = batch["lowpoly"].to(device)
            
            # Forward pass
            target_vertex_count = lowpoly.pos.shape[0]
            
            if USE_NEW_MODEL:
                pred_positions, pred_features, importance_scores, selected_indices = model(
                    highpoly, target_vertex_count=target_vertex_count
                )
                # Filtrar edge_index para usar apenas vértices selecionados
                n_original = highpoly.edge_index.max().item() + 1 if highpoly.edge_index.numel() > 0 else 0
                if n_original > 0:
                    mapping = torch.full((n_original,), -1, dtype=torch.long, device=highpoly.edge_index.device)
                    mapping[selected_indices] = torch.arange(len(selected_indices), device=highpoly.edge_index.device)
                    
                    mask = (mapping[highpoly.edge_index[0]] >= 0) & (mapping[highpoly.edge_index[1]] >= 0)
                    filtered_edges = highpoly.edge_index[:, mask]
                    
                    if filtered_edges.numel() > 0:
                        filtered_edges[0] = mapping[filtered_edges[0]]
                        filtered_edges[1] = mapping[filtered_edges[1]]
                        pred_edge_index = filtered_edges
                    else:
                        pred_edge_index = torch.empty((2, 0), dtype=torch.long, device=highpoly.edge_index.device)
                else:
                    pred_edge_index = torch.empty((2, 0), dtype=torch.long, device=highpoly.edge_index.device)
            else:
                pred_positions, pred_features = model(highpoly)
                pred_edge_index = highpoly.edge_index
            
            # Calcular loss
            target_positions = lowpoly.pos
            target_features = lowpoly.x[:, 3:9] if lowpoly.x.shape[1] >= 9 else None
            
            # Preparar argumentos da loss (compatível com versão antiga e nova)
            loss_args = {
                "pred_positions": pred_positions,
                "pred_features": pred_features,
                "target_positions": target_positions,
                "target_features": target_features
            }
            
            # Adicionar argumentos extras se usar nova loss
            if USE_NEW_LOSS:
                loss_args.update({
                    "pred_edge_index": pred_edge_index,
                    "target_edge_index": lowpoly.edge_index,
                    "target_vertex_count": target_vertex_count
                })
            
            loss, loss_dict = criterion(**loss_args)
            
            # Acumular estatísticas
            total_loss += loss.item()
            for key in loss_details:
                loss_details[key] += loss_dict.get(key, 0.0)
            num_batches += 1
            
            pos_loss_key = 'chamfer_loss' if 'chamfer_loss' in loss_dict else 'pos_loss'
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "pos": f"{loss_dict.get(pos_loss_key, 0):.4f}"
            })
    
    # Médias
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    for key in loss_details:
        loss_details[key] /= num_batches if num_batches > 0 else 1.0
    
    return avg_loss, loss_details


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, device="cpu"):
    """
    Carrega um checkpoint do modelo.
    
    Args:
        checkpoint_path: Caminho para o checkpoint
        model: Modelo a ser carregado
        optimizer: Otimizador (opcional)
        scheduler: Scheduler (opcional)
        device: Dispositivo
        
    Returns:
        epoch: Época do checkpoint
        best_val_loss: Melhor loss de validação (se disponível)
        history: Histórico de treinamento (se disponível)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Carregar estado do modelo
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # Carregar estado do otimizador
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    # Carregar estado do scheduler (se disponível)
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    epoch = checkpoint.get("epoch", 0)
    best_val_loss = checkpoint.get("best_val_loss", float('inf'))
    history = checkpoint.get("history", {"train_loss": [], "val_loss": [], "epochs": []})
    
    print(f"✓ Checkpoint carregado: época {epoch}")
    if best_val_loss != float('inf'):
        print(f"✓ Melhor loss de validação: {best_val_loss:.6f}")
    
    return epoch, best_val_loss, history


def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir, is_best=False, scheduler=None, best_val_loss=None, history=None):
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
    
    # Adicionar scheduler se disponível
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    
    # Adicionar melhor loss e histórico
    if best_val_loss is not None:
        checkpoint["best_val_loss"] = best_val_loss
    if history is not None:
        checkpoint["history"] = history
    
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
    parser = argparse.ArgumentParser(description="Treinar modelo GNN para retopologia")
    parser.add_argument(
        "--resume", "-r",
        type=str,
        default=None,
        help="Caminho para checkpoint para continuar treinamento (ex: checkpoints/checkpoint_epoch_50.pt ou checkpoints/last_checkpoint.pt)"
    )
    args = parser.parse_args()
    
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
    if USE_NEW_MODEL:
        model = RetopologyGNN(MODEL_CONFIG, target_vertex_count=None).to(device)
        print("✓ Usando modelo estável (StableRetopologyGNN)")
    else:
        model = RetopologyGNN(MODEL_CONFIG).to(device)
        print("✓ Usando modelo original (compatibilidade)")
    
    print(f"✓ Modelo criado com {sum(p.numel() for p in model.parameters())} parâmetros")
    print()
    
    # Loss e otimizador
    if USE_NEW_LOSS:
        from loss_v2 import ImprovedRetopologyLoss
        criterion = ImprovedRetopologyLoss()
        print("✓ Usando loss melhorada (ImprovedRetopologyLoss)")
    else:
        criterion = RetopologyLoss()
        print("✓ Usando loss original (compatibilidade)")
    print()
    optimizer = optim.Adam(model.parameters(), lr=TRAINING_CONFIG["learning_rate"])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # Histórico de treinamento
    history = {
        "train_loss": [],
        "val_loss": [],
        "epochs": []
    }
    
    best_val_loss = float('inf')
    start_epoch = 1
    start_time = time.time()
    
    # Carregar checkpoint se especificado
    if args.resume:
        checkpoint_path = Path(args.resume)
        if not checkpoint_path.exists():
            print(f"❌ Erro: Checkpoint não encontrado: {checkpoint_path}")
            return
        
        print(f"Carregando checkpoint de {checkpoint_path}...")
        start_epoch, best_val_loss, loaded_history = load_checkpoint(
            checkpoint_path, model, optimizer, scheduler, device
        )
        
        # Usar histórico carregado se disponível
        if loaded_history and loaded_history.get("train_loss"):
            history = loaded_history
            print(f"✓ Histórico carregado: {len(history['train_loss'])} épocas anteriores")
        else:
            # Tentar carregar histórico do arquivo JSON
            history_path = CHECKPOINTS_DIR / "training_history.json"
            if history_path.exists():
                try:
                    with open(history_path, 'r') as f:
                        history = json.load(f)
                    print(f"✓ Histórico carregado do arquivo JSON: {len(history.get('train_loss', []))} épocas")
                except Exception as e:
                    print(f"⚠️  Aviso: Não foi possível carregar histórico: {e}")
        
        start_epoch += 1  # Começar da próxima época
        print(f"✓ Continuando treinamento da época {start_epoch}")
        print()
    
    # Loop de treinamento
    print("Iniciando treinamento...")
    print("=" * 70)
    
    for epoch in range(start_epoch, TRAINING_CONFIG["num_epochs"] + 1):
        print(f"\nÉpoca {epoch}/{TRAINING_CONFIG['num_epochs']}")
        print("-" * 70)
        
        # Treinar
        train_loss, train_details = train_epoch(model, train_loader, criterion, optimizer, device)
        history["train_loss"].append(train_loss)
        
        print(f"Loss de treino: {train_loss:.6f}")
        print(f"  - Chamfer: {train_details.get('chamfer_loss', train_details.get('pos_loss', 0)):.6f}")
        if 'hausdorff_loss' in train_details:
            print(f"  - Hausdorff: {train_details['hausdorff_loss']:.6f}")
        if 'edge_length_loss' in train_details:
            print(f"  - Edge Length: {train_details['edge_length_loss']:.6f}")
        if 'degenerate_loss' in train_details:
            print(f"  - Degenerate: {train_details['degenerate_loss']:.6f}")
        if 'smoothness_loss' in train_details:
            print(f"  - Smoothness: {train_details['smoothness_loss']:.6f}")
        if 'vertex_count_loss' in train_details:
            print(f"  - Vertex Count: {train_details['vertex_count_loss']:.6f}")
        print(f"  - Features: {train_details.get('feature_loss', 0):.6f}")
        if 'normal_consistency_loss' in train_details:
            print(f"  - Normal Consistency: {train_details['normal_consistency_loss']:.6f}")
        
        # Validar
        if epoch % TRAINING_CONFIG["val_every"] == 0:
            val_loss, val_details = validate(model, val_loader, criterion, device)
            history["val_loss"].append(val_loss)
            history["epochs"].append(epoch)
            
            print(f"Loss de validação: {val_loss:.6f}")
            print(f"  - Chamfer: {val_details.get('chamfer_loss', val_details.get('pos_loss', 0)):.6f}")
            if 'hausdorff_loss' in val_details:
                print(f"  - Hausdorff: {val_details['hausdorff_loss']:.6f}")
            if 'edge_length_loss' in val_details:
                print(f"  - Edge Length: {val_details['edge_length_loss']:.6f}")
            if 'degenerate_loss' in val_details:
                print(f"  - Degenerate: {val_details['degenerate_loss']:.6f}")
            if 'smoothness_loss' in val_details:
                print(f"  - Smoothness: {val_details['smoothness_loss']:.6f}")
            if 'vertex_count_loss' in val_details:
                print(f"  - Vertex Count: {val_details['vertex_count_loss']:.6f}")
            print(f"  - Features: {val_details.get('feature_loss', 0):.6f}")
            if 'normal_consistency_loss' in val_details:
                print(f"  - Normal Consistency: {val_details['normal_consistency_loss']:.6f}")
            
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
            save_checkpoint(
                model, optimizer, epoch, train_loss, CHECKPOINTS_DIR,
                is_best=False, scheduler=scheduler, best_val_loss=best_val_loss, history=history
            )
            print(f"✓ Checkpoint salvo (época {epoch})")
        
        # Salvar histórico
        history_path = CHECKPOINTS_DIR / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
    
    # Salvar checkpoint final
    save_checkpoint(
        model, optimizer, TRAINING_CONFIG["num_epochs"], train_loss, CHECKPOINTS_DIR,
        is_best=False, scheduler=scheduler, best_val_loss=best_val_loss, history=history
    )
    
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

