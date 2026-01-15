"""
Modelo GNN para retopologia: converte high poly mesh em low poly mesh.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch


class RetopologyGNN(nn.Module):
    """
    Modelo GNN para retopologia usando arquitetura encoder-decoder.
    """
    
    def __init__(self, config):
        super(RetopologyGNN, self).__init__()
        
        self.node_features = config["node_features"]
        self.hidden_dim = config["hidden_dim"]
        self.num_layers = config["num_layers"]
        self.dropout = config["dropout"]
        self.use_batch_norm = config["use_batch_norm"]
        
        # Encoder: processa o high poly mesh
        self.encoder_convs = nn.ModuleList()
        self.encoder_bns = nn.ModuleList() if self.use_batch_norm else None
        
        # Primeira camada
        self.encoder_convs.append(GCNConv(self.node_features, self.hidden_dim))
        if self.use_batch_norm:
            self.encoder_bns.append(nn.BatchNorm1d(self.hidden_dim))
        
        # Camadas intermediárias
        for _ in range(self.num_layers - 2):
            self.encoder_convs.append(GCNConv(self.hidden_dim, self.hidden_dim))
            if self.use_batch_norm:
                self.encoder_bns.append(nn.BatchNorm1d(self.hidden_dim))
        
        # Última camada do encoder
        self.encoder_convs.append(GCNConv(self.hidden_dim, self.hidden_dim))
        if self.use_batch_norm:
            self.encoder_bns.append(nn.BatchNorm1d(self.hidden_dim))
        
        # Decoder: gera o low poly mesh
        # Primeiro, reduz o número de vértices (pooling)
        self.vertex_reduction = nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        
        # Decoder layers
        self.decoder_convs = nn.ModuleList()
        self.decoder_bns = nn.ModuleList() if self.use_batch_norm else None
        
        for _ in range(self.num_layers - 1):
            self.decoder_convs.append(GCNConv(self.hidden_dim // 2, self.hidden_dim // 2))
            if self.use_batch_norm:
                self.decoder_bns.append(nn.BatchNorm1d(self.hidden_dim // 2))
        
        # Output: gera posições dos vértices do low poly
        self.output_conv = GCNConv(self.hidden_dim // 2, 3)  # 3D coordinates
        
        # Feature decoder para outras features (normais, etc.)
        self.feature_decoder = nn.Sequential(
            nn.Linear(self.hidden_dim // 2, self.hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 4, 6)  # normal (3) + outras features (3)
        )
    
    def forward(self, data):
        """
        Forward pass do modelo.
        
        Args:
            data: Batch de grafos PyTorch Geometric
            
        Returns:
            low_poly_positions: Posições dos vértices do low poly
            low_poly_features: Features adicionais (normais, etc.)
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Encoder
        for i, conv in enumerate(self.encoder_convs):
            x = conv(x, edge_index)
            if self.use_batch_norm:
                x = self.encoder_bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Redução de vértices (simulação de decimação)
        # Usamos um mecanismo de atenção para selecionar vértices importantes
        x_reduced = self.vertex_reduction(x)
        x_reduced = F.relu(x_reduced)
        
        # Decoder
        for i, conv in enumerate(self.decoder_convs):
            x_reduced = conv(x_reduced, edge_index)
            if self.use_batch_norm:
                x_reduced = self.decoder_bns[i](x_reduced)
            x_reduced = F.relu(x_reduced)
            x_reduced = F.dropout(x_reduced, p=self.dropout, training=self.training)
        
        # Output: posições dos vértices
        low_poly_positions = self.output_conv(x_reduced, edge_index)
        
        # Features adicionais
        low_poly_features = self.feature_decoder(x_reduced)
        
        return low_poly_positions, low_poly_features
    
    def predict_mesh(self, high_poly_data, target_num_vertices=None):
        """
        Gera um low poly mesh a partir de um high poly mesh.
        
        Args:
            high_poly_data: Data object do high poly mesh
            target_num_vertices: Número desejado de vértices (opcional)
            
        Returns:
            low_poly_data: Data object do low poly mesh gerado
        """
        self.eval()
        with torch.no_grad():
            low_poly_pos, low_poly_feat = self.forward(high_poly_data)
            
            # Se especificado, reduzir para número de vértices alvo
            if target_num_vertices and low_poly_pos.shape[0] > target_num_vertices:
                # Selecionar os vértices mais importantes (baseado na magnitude)
                importance = torch.norm(low_poly_pos, dim=1)
                _, indices = torch.topk(importance, target_num_vertices)
                low_poly_pos = low_poly_pos[indices]
                low_poly_feat = low_poly_feat[indices]
            
            # Criar novo Data object
            low_poly_data = Data(
                x=torch.cat([low_poly_pos, low_poly_feat], dim=1),
                pos=low_poly_pos,
                edge_index=high_poly_data.edge_index  # Manter mesma topologia inicialmente
            )
            
            return low_poly_data


class ImprovedRetopologyGNN(nn.Module):
    """
    Versão melhorada do modelo com atenção e pooling adaptativo.
    """
    
    def __init__(self, config):
        super(ImprovedRetopologyGNN, self).__init__()
        
        self.node_features = config["node_features"]
        self.hidden_dim = config["hidden_dim"]
        self.num_layers = config["num_layers"]
        self.dropout = config["dropout"]
        
        # Encoder com GAT (Graph Attention)
        self.encoder_layers = nn.ModuleList()
        self.encoder_layers.append(GATConv(self.node_features, self.hidden_dim, heads=4, dropout=self.dropout))
        
        for _ in range(self.num_layers - 2):
            self.encoder_layers.append(GATConv(self.hidden_dim * 4, self.hidden_dim, heads=4, dropout=self.dropout))
        
        self.encoder_layers.append(GATConv(self.hidden_dim * 4, self.hidden_dim, heads=1, dropout=self.dropout))
        
        # Mecanismo de pooling adaptativo
        self.attention_pool = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Decoder
        self.decoder_layers = nn.ModuleList()
        for _ in range(self.num_layers - 1):
            self.decoder_layers.append(GCNConv(self.hidden_dim, self.hidden_dim))
        
        # Output
        self.position_head = nn.Linear(self.hidden_dim, 3)
        self.feature_head = nn.Linear(self.hidden_dim, 6)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # Encoder
        for i, layer in enumerate(self.encoder_layers):
            x = layer(x, edge_index)
            if i < len(self.encoder_layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Pooling adaptativo com atenção
        attention_weights = self.attention_pool(x)
        x_pooled = x * attention_weights
        
        # Decoder
        for layer in self.decoder_layers:
            x_pooled = layer(x_pooled, edge_index)
            x_pooled = F.relu(x_pooled)
            x_pooled = F.dropout(x_pooled, p=self.dropout, training=self.training)
        
        # Outputs
        positions = self.position_head(x_pooled)
        features = self.feature_head(x_pooled)
        
        return positions, features

