"""
Modelo GNN para ModelRetopology2000: transforma highpoly em lowpoly usando
heatmap (importância), highpoly e par lowpoly como referência.
Entrada: grafo highpoly com 10 features (9 + importância do heatmap).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Batch


class RetopologyGNN2000(nn.Module):
    """
    GNN encoder-decoder que recebe highpoly + importância (heatmap) e produz posições lowpoly.
    """

    def __init__(self, config):
        super(RetopologyGNN2000, self).__init__()
        self.node_features = config["node_features"]
        self.hidden_dim = config["hidden_dim"]
        self.num_layers = config["num_layers"]
        self.dropout = config["dropout"]
        self.use_batch_norm = config["use_batch_norm"]

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

        self.vertex_reduction = nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        self.decoder_convs = nn.ModuleList()
        self.decoder_bns = nn.ModuleList() if self.use_batch_norm else None
        for _ in range(self.num_layers - 1):
            self.decoder_convs.append(GCNConv(self.hidden_dim // 2, self.hidden_dim // 2))
            if self.use_batch_norm:
                self.decoder_bns.append(nn.BatchNorm1d(self.hidden_dim // 2))
        self.output_conv = GCNConv(self.hidden_dim // 2, 3)
        self.feature_decoder = nn.Sequential(
            nn.Linear(self.hidden_dim // 2, self.hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 4, 6),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for i, conv in enumerate(self.encoder_convs):
            x = conv(x, edge_index)
            if self.use_batch_norm:
                x = self.encoder_bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x_reduced = self.vertex_reduction(x)
        x_reduced = F.relu(x_reduced)
        for i, conv in enumerate(self.decoder_convs):
            x_reduced = conv(x_reduced, edge_index)
            if self.use_batch_norm:
                x_reduced = self.decoder_bns[i](x_reduced)
            x_reduced = F.relu(x_reduced)
            x_reduced = F.dropout(x_reduced, p=self.dropout, training=self.training)
        low_poly_positions = self.output_conv(x_reduced, edge_index)
        low_poly_features = self.feature_decoder(x_reduced)
        return low_poly_positions, low_poly_features
