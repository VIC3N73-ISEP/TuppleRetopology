"""
Loss estrutural para retopologia: preserva forma e conectividade.
Chamfer + Hausdorff + Edge Length + Laplacian + Normal Consistency.
"""

import torch
import torch.nn as nn


def chamfer_distance(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """pred (N,3), target (M,3)."""
    d = torch.cdist(pred, target)
    return d.min(dim=1)[0].mean() + d.min(dim=0)[0].mean()


def hausdorff_distance(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Distância máxima entre conjuntos."""
    d = torch.cdist(pred, target)
    d1 = d.min(dim=1)[0].max()
    d2 = d.min(dim=0)[0].max()
    return torch.max(d1, d2)


def edge_length_loss(pos: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Regularização: variância dos comprimentos das arestas (evita arestas muito longas/soltas)."""
    if edge_index.shape[1] == 0:
        return pos.new_tensor(0.0)
    row, col = edge_index[0], edge_index[1]
    lengths = torch.norm(pos[row] - pos[col], dim=1)
    return torch.var(lengths) + 1e-6


def laplacian_smoothness_loss(pos: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Laplacian: v_i - mean(vizinhos). Evita spikes e mantém suavidade."""
    if edge_index.shape[1] == 0:
        return pos.new_tensor(0.0)
    row, col = edge_index[0], edge_index[1]
    n = pos.shape[0]
    deg = torch.zeros(n, device=pos.device, dtype=pos.dtype)
    deg.scatter_add_(0, row, torch.ones(edge_index.shape[1], device=pos.device, dtype=pos.dtype))
    deg = deg.clamp(min=1e-6).unsqueeze(1)
    neighbor_sum = torch.zeros_like(pos)
    neighbor_sum.scatter_add_(0, row.unsqueeze(-1).expand(-1, 3), pos[col])
    lap = pos - neighbor_sum / deg
    return torch.norm(lap, dim=1).mean()


def normal_consistency_loss(
    pred_features: torch.Tensor,
    target_features: torch.Tensor,
) -> torch.Tensor:
    """Consistência de normais (primeiros 3 dims de features). Usa Chamfer em features se tamanhos diferentes."""
    if pred_features.shape[0] == 0 or target_features.shape[0] == 0:
        return pred_features.new_tensor(0.0)
    n_pred = pred_features[:, :3]
    n_tgt = target_features[:, :3]
    if n_pred.shape[0] == n_tgt.shape[0]:
        return nn.functional.mse_loss(n_pred, n_tgt)
    d = torch.cdist(n_pred, n_tgt)
    return d.min(dim=1)[0].mean() + d.min(dim=0)[0].mean()


class RetopologyStructuralLoss(nn.Module):
    """
    Loss combinada para retopologia:
    - Chamfer + Hausdorff (forma global)
    - Edge length + Laplacian (estrutura, evita vértices soltos)
    - Normal consistency
    Pesos configuráveis.
    """

    def __init__(
        self,
        chamfer_weight=1.0,
        hausdorff_weight=0.5,
        edge_weight=0.2,
        laplacian_weight=0.2,
        normal_weight=0.3,
    ):
        super().__init__()
        self.chamfer_weight = chamfer_weight
        self.hausdorff_weight = hausdorff_weight
        self.edge_weight = edge_weight
        self.laplacian_weight = laplacian_weight
        self.normal_weight = normal_weight

    def forward(
        self,
        pred_positions: torch.Tensor,
        pred_features: torch.Tensor,
        target_positions: torch.Tensor,
        target_features: torch.Tensor = None,
        pred_edge_index: torch.Tensor = None,
        target_edge_index: torch.Tensor = None,
        **kwargs,
    ):
        """
        pred_positions (N,3), pred_features (N,F), target_* (M,...).
        pred_edge_index: arestas do grafo reduzido (para edge + laplacian).
        """
        loss_dict = {}
        total = pred_positions.new_tensor(0.0)

        c = chamfer_distance(pred_positions, target_positions)
        total = total + self.chamfer_weight * c
        loss_dict["chamfer_loss"] = c.item()

        h = hausdorff_distance(pred_positions, target_positions)
        total = total + self.hausdorff_weight * h
        loss_dict["hausdorff_loss"] = h.item()

        if pred_edge_index is not None and pred_edge_index.shape[1] > 0:
            e = edge_length_loss(pred_positions, pred_edge_index)
            total = total + self.edge_weight * e
            loss_dict["edge_length_loss"] = e.item()
            lap = laplacian_smoothness_loss(pred_positions, pred_edge_index)
            total = total + self.laplacian_weight * lap
            loss_dict["laplacian_loss"] = lap.item()
        else:
            loss_dict["edge_length_loss"] = 0.0
            loss_dict["laplacian_loss"] = 0.0

        if target_features is not None and pred_features.shape[1] >= 3:
            nloss = normal_consistency_loss(pred_features, target_features)
            total = total + self.normal_weight * nloss
            loss_dict["normal_consistency_loss"] = nloss.item()
        else:
            loss_dict["normal_consistency_loss"] = 0.0

        loss_dict["total_loss"] = total.item()
        return total, loss_dict
