"""
ModelRetopology3000: modelo de previsão de ratio para decimate-only.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool


class RatioHead(nn.Module):
    """
    Encoder GCN curto sobre o high (9 features) + pooling global + MLP -> r_hat.
    """

    def __init__(self, config):
        super().__init__()
        self.base_dim = config["base_node_features"]
        h = config["hidden_dim"]
        n_layers = max(1, int(config.get("ratio_gcn_layers", 3)))
        self.ratio_convs = nn.ModuleList()
        self.ratio_bns = nn.ModuleList() if config["use_batch_norm"] else None
        self.ratio_convs.append(GCNConv(self.base_dim, h))
        if config["use_batch_norm"]:
            self.ratio_bns.append(nn.BatchNorm1d(h))
        for _ in range(n_layers - 1):
            self.ratio_convs.append(GCNConv(h, h))
            if config["use_batch_norm"]:
                self.ratio_bns.append(nn.BatchNorm1d(h))
        self.dropout = config["dropout"]
        pool_dim = 2 * h
        self.ratio_mlp = nn.Sequential(
            nn.Linear(pool_dim, h),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(h, 1),
        )

    def forward(self, data):
        x = data.x[:, : self.base_dim]
        edge_index = data.edge_index
        batch = data.batch
        for i, conv in enumerate(self.ratio_convs):
            x = conv(x, edge_index)
            if self.ratio_bns is not None:
                x = self.ratio_bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        g = torch.cat([x_mean, x_max], dim=1)
        logits = self.ratio_mlp(g).squeeze(-1)
        r_hat = torch.sigmoid(logits)
        return r_hat


class RetopologyModel3000(nn.Module):
    """Modelo oficial do 3000: previsão de ratio para decimate-only."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.ratio_head = RatioHead(config)
    def predict_ratio(self, high_batch):
        return self.ratio_head(high_batch)

    def forward(self, high_batch):
        return self.predict_ratio(high_batch)
