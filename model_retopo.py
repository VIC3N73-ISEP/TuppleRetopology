"""
Modelo de retopologia com seleção diferenciável: importance head + Top-K.
Reduz para exatamente K vértices mantendo estrutura.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Batch


def _filter_edge_index_to_selection(edge_index, selected_indices, num_nodes):
    """Mantém apenas arestas cujos dois extremos estão em selected_indices; reindexa para 0..K-1."""
    device = edge_index.device
    mapping = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
    mapping[selected_indices] = torch.arange(len(selected_indices), device=device)
    row, col = edge_index[0], edge_index[1]
    mask = (mapping[row] >= 0) & (mapping[col] >= 0)
    if not mask.any():
        return torch.empty((2, 0), dtype=torch.long, device=device)
    new_row = mapping[row[mask]]
    new_col = mapping[col[mask]]
    return torch.stack([new_row, new_col], dim=0)


class RetopologyGNN2000(nn.Module):
    """
    GNN para retopologia com número fixo de vértices (ex.: 2000).
    - Backbone GNN → features por vértice
    - Position head → novas posições
    - Importance head → score por vértice → Top-K seleção
    """

    def __init__(self, config, target_vertices=2000):
        super().__init__()
        self.node_features = config["node_features"]
        self.hidden_dim = config["hidden_dim"]
        self.num_layers = config["num_layers"]
        self.dropout = config["dropout"]
        self.use_batch_norm = config.get("use_batch_norm", True)
        self.target_vertices = target_vertices

        # Backbone encoder (GCN)
        self.encoder_convs = nn.ModuleList()
        self.encoder_bns = nn.ModuleList() if self.use_batch_norm else None

        self.encoder_convs.append(GCNConv(self.node_features, self.hidden_dim))
        if self.use_batch_norm:
            self.encoder_bns.append(nn.BatchNorm1d(self.hidden_dim))

        for _ in range(self.num_layers - 2):
            self.encoder_convs.append(GCNConv(self.hidden_dim, self.hidden_dim))
            if self.use_batch_norm:
                self.encoder_bns.append(nn.BatchNorm1d(self.hidden_dim))

        self.encoder_convs.append(GCNConv(self.hidden_dim, self.hidden_dim))
        if self.use_batch_norm:
            self.encoder_bns.append(nn.BatchNorm1d(self.hidden_dim))

        # Heads
        self.position_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 3),
        )
        self.importance_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 1),
        )
        self.feature_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, 6),
        )

    def forward(self, data, use_topk=True, target_vertices=None):
        """
        Args:
            data: Batch PyG (x, edge_index, batch)
            use_topk: Se True, aplica Top-K e devolve exatamente K vértices
            target_vertices: K desejado (usa self.target_vertices se None)

        Returns:
            pred_positions, pred_features, importance_scores, selected_indices, pred_edge_index
            - selected_indices e pred_edge_index são None quando use_topk=False
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        num_nodes = x.shape[0]

        # Backbone
        for i, conv in enumerate(self.encoder_convs):
            x = conv(x, edge_index)
            if self.use_batch_norm:
                x = self.encoder_bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Predições por vértice
        pred_positions = self.position_head(x)
        importance_scores = torch.sigmoid(self.importance_head(x)).squeeze(-1)
        pred_features = self.feature_head(x)

        selected_indices = None
        pred_edge_index = edge_index  # Fase 1 usa o grafo completo

        if use_topk and target_vertices is not None:
            K = min(int(target_vertices), num_nodes)
            if K < num_nodes:
                _, selected_indices = torch.topk(importance_scores, K, dim=0)
                pred_positions = pred_positions[selected_indices]
                pred_features = pred_features[selected_indices]
                pred_edge_index = _filter_edge_index_to_selection(
                    edge_index, selected_indices, num_nodes
                )
            else:
                selected_indices = torch.arange(num_nodes, device=x.device)
                pred_edge_index = edge_index

        return pred_positions, pred_features, importance_scores, selected_indices, pred_edge_index
