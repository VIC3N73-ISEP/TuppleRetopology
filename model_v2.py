"""
Modelo GNN refatorado para retopologia com controle explícito de complexidade.
Abordagem: Vertex Importance Score + Top-K Selection (Abordagem B)

Mudanças principais:
1. Vertex importance score explícito e diferenciável
2. Top-K selection durante treinamento (com Gumbel-Softmax para diferenciabilidade)
3. Vertex budget fixo e controlável
4. Melhor estrutura encoder-decoder com bottleneck explícito
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data


class StableRetopologyGNN(nn.Module):
    """
    Modelo GNN estável para retopologia com controle explícito de vertex budget.
    
    Arquitetura:
    - Encoder: Processa high poly mesh
    - Vertex Importance Head: Calcula scores de importância para cada vértice
    - Top-K Selection: Seleciona K vértices mais importantes (diferenciável)
    - Decoder: Gera posições e features dos vértices selecionados
    """
    
    def __init__(self, config, target_vertex_count=None):
        super(StableRetopologyGNN, self).__init__()
        
        self.node_features = config["node_features"]
        self.hidden_dim = config["hidden_dim"]
        self.num_layers = config["num_layers"]
        self.dropout = config["dropout"]
        self.use_batch_norm = config["use_batch_norm"]
        
        # Target vertex count (pode ser None para inferência flexível)
        self.target_vertex_count = target_vertex_count
        
        # ========== ENCODER ==========
        # Processa o high poly mesh e extrai features hierárquicas
        self.encoder_convs = nn.ModuleList()
        self.encoder_bns = nn.ModuleList() if self.use_batch_norm else None
        
        # Primeira camada: entrada → hidden
        self.encoder_convs.append(GCNConv(self.node_features, self.hidden_dim))
        if self.use_batch_norm:
            self.encoder_bns.append(nn.BatchNorm1d(self.hidden_dim))
        
        # Camadas intermediárias: hidden → hidden
        for _ in range(self.num_layers - 2):
            self.encoder_convs.append(GCNConv(self.hidden_dim, self.hidden_dim))
            if self.use_batch_norm:
                self.encoder_bns.append(nn.BatchNorm1d(self.hidden_dim))
        
        # Última camada do encoder
        self.encoder_convs.append(GCNConv(self.hidden_dim, self.hidden_dim))
        if self.use_batch_norm:
            self.encoder_bns.append(nn.BatchNorm1d(self.hidden_dim))
        
        # ========== VERTEX IMPORTANCE HEAD ==========
        # Calcula um score de importância para cada vértice
        # Score alto = vértice deve ser mantido no low poly
        self.importance_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 4, 1),  # Score único por vértice
            nn.Sigmoid()  # Score entre 0 e 1
        )
        
        # ========== DECODER ==========
        # Gera posições e features para os vértices selecionados
        # Reduz dimensão para o bottleneck (simula redução de vértices)
        self.bottleneck_dim = self.hidden_dim // 2
        
        # Camadas do decoder
        self.decoder_convs = nn.ModuleList()
        self.decoder_bns = nn.ModuleList() if self.use_batch_norm else None
        
        # Primeira camada: hidden → bottleneck
        self.decoder_convs.append(GCNConv(self.hidden_dim, self.bottleneck_dim))
        if self.use_batch_norm:
            self.decoder_bns.append(nn.BatchNorm1d(self.bottleneck_dim))
        
        # Camadas intermediárias: bottleneck → bottleneck
        for _ in range(self.num_layers - 2):
            self.decoder_convs.append(GCNConv(self.bottleneck_dim, self.bottleneck_dim))
            if self.use_batch_norm:
                self.decoder_bns.append(nn.BatchNorm1d(self.bottleneck_dim))
        
        # ========== OUTPUT HEADS ==========
        # Head para posições 3D
        self.position_head = GCNConv(self.bottleneck_dim, 3)
        
        # Head para features (normais, etc.)
        self.feature_head = nn.Sequential(
            nn.Linear(self.bottleneck_dim, self.bottleneck_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.bottleneck_dim // 2, 6)  # normal (3) + extras (3)
        )
    
    def forward(self, data, target_vertex_count=None, temperature=1.0):
        """
        Forward pass do modelo.
        
        Args:
            data: Batch de grafos PyTorch Geometric
            target_vertex_count: Número desejado de vértices (None = usar self.target_vertex_count)
            temperature: Temperatura para Gumbel-Softmax (menor = mais hard selection)
            
        Returns:
            low_poly_positions: Posições dos vértices do low poly (K, 3)
            low_poly_features: Features dos vértices (K, 6)
            importance_scores: Scores de importância originais (N,)
            selected_indices: Índices dos vértices selecionados (K,)
        """
        x, edge_index = data.x, data.edge_index
        
        # ========== ENCODER ==========
        for i, conv in enumerate(self.encoder_convs):
            x = conv(x, edge_index)
            if self.use_batch_norm:
                x = self.encoder_bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # ========== VERTEX IMPORTANCE ==========
        # Calcular scores de importância para cada vértice
        importance_scores = self.importance_head(x).squeeze(-1)  # (N,)
        
        # Verificar e corrigir NaN/Inf
        importance_scores = torch.nan_to_num(importance_scores, nan=0.0, posinf=1.0, neginf=0.0)
        importance_scores = torch.clamp(importance_scores, min=0.0, max=1.0)
        
        # ========== TOP-K SELECTION (Diferenciável) ==========
        # Usar Gumbel-Softmax para tornar a seleção diferenciável
        K = target_vertex_count if target_vertex_count is not None else self.target_vertex_count
        
        if K is None:
            # Se não especificado, usar todos os vértices (modo compatibilidade)
            selected_indices = torch.arange(x.shape[0], device=x.device)
            selection_weights = torch.ones_like(importance_scores)
        else:
            # Garantir que K é válido
            K = min(K, x.shape[0])
            if K <= 0:
                K = 1
            
            # Seleção diferenciável usando Gumbel-Softmax
            selected_indices, selection_weights = self._differentiable_topk(
                importance_scores, K, temperature
            )
        
        # Aplicar pesos de seleção às features
        x_selected = x[selected_indices] * selection_weights.unsqueeze(-1)
        
        # Verificar se há NaN ou Inf nas features
        if torch.isnan(x_selected).any() or torch.isinf(x_selected).any():
            x_selected = torch.nan_to_num(x_selected, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # ========== DECODER ==========
        # Criar novo edge_index apenas para vértices selecionados
        edge_index_selected = self._filter_edges(edge_index, selected_indices)
        
        # Decoder
        if edge_index_selected.shape[1] > 0:
            # Usar GCNConv quando há arestas
            for i, conv in enumerate(self.decoder_convs):
                x_selected = conv(x_selected, edge_index_selected)
                if self.use_batch_norm:
                    x_selected = self.decoder_bns[i](x_selected)
                x_selected = F.relu(x_selected)
                x_selected = F.dropout(x_selected, p=self.dropout, training=self.training)
                
                # Verificar NaN após cada camada
                if torch.isnan(x_selected).any() or torch.isinf(x_selected).any():
                    x_selected = torch.nan_to_num(x_selected, nan=0.0, posinf=1e6, neginf=-1e6)
                    x_selected = torch.clamp(x_selected, min=-1e6, max=1e6)
        else:
            # Quando não há arestas, criar um grafo completo temporário para o decoder
            # Isso permite usar as mesmas camadas GCNConv
            num_selected = x_selected.shape[0]
            if num_selected > 1:
                # Criar grafo completo (fully connected)
                row = torch.arange(num_selected, device=x_selected.device).repeat(num_selected)
                col = torch.arange(num_selected, device=x_selected.device).repeat_interleave(num_selected)
                # Remover self-loops
                mask = row != col
                edge_index_selected = torch.stack([row[mask], col[mask]], dim=0)
            else:
                # Se há apenas 1 vértice, criar self-loop
                edge_index_selected = torch.zeros((2, 1), dtype=torch.long, device=x_selected.device)
            
            # Agora processar com GCNConv normalmente
            for i, conv in enumerate(self.decoder_convs):
                x_selected = conv(x_selected, edge_index_selected)
                if self.use_batch_norm:
                    x_selected = self.decoder_bns[i](x_selected)
                x_selected = F.relu(x_selected)
                x_selected = F.dropout(x_selected, p=self.dropout, training=self.training)
                
                # Verificar NaN após cada camada
                if torch.isnan(x_selected).any() or torch.isinf(x_selected).any():
                    x_selected = torch.nan_to_num(x_selected, nan=0.0, posinf=1e6, neginf=-1e6)
                    x_selected = torch.clamp(x_selected, min=-1e6, max=1e6)
        
        # ========== OUTPUT ==========
        # Posições 3D
        # Se não há arestas após o decoder, criar grafo completo novamente
        if edge_index_selected.shape[1] == 0:
            num_selected = x_selected.shape[0]
            if num_selected > 1:
                row = torch.arange(num_selected, device=x_selected.device).repeat(num_selected)
                col = torch.arange(num_selected, device=x_selected.device).repeat_interleave(num_selected)
                mask = row != col
                edge_index_selected = torch.stack([row[mask], col[mask]], dim=0)
            else:
                edge_index_selected = torch.zeros((2, 1), dtype=torch.long, device=x_selected.device)
        
        low_poly_positions = self.position_head(x_selected, edge_index_selected)
        
        # Features adicionais
        low_poly_features = self.feature_head(x_selected)
        
        # Verificar e corrigir NaN/Inf nos outputs
        low_poly_positions = torch.nan_to_num(low_poly_positions, nan=0.0, posinf=1e6, neginf=-1e6)
        low_poly_features = torch.nan_to_num(low_poly_features, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Clamp valores extremos
        low_poly_positions = torch.clamp(low_poly_positions, min=-10.0, max=10.0)
        low_poly_features = torch.clamp(low_poly_features, min=-10.0, max=10.0)
        
        return low_poly_positions, low_poly_features, importance_scores, selected_indices
    
    def _differentiable_topk(self, scores, k, temperature=1.0):
        """
        Seleção top-K diferenciável usando Gumbel-Softmax.
        
        Args:
            scores: Scores de importância (N,)
            k: Número de vértices a selecionar
            temperature: Temperatura (menor = mais hard)
            
        Returns:
            indices: Índices selecionados (k,)
            weights: Pesos de seleção (k,)
        """
        n = scores.shape[0]
        k = min(k, n)  # Não pode selecionar mais que o total
        
        if self.training:
            # Durante treinamento: usar Gumbel-Softmax para diferenciabilidade
            # Adicionar ruído Gumbel
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(scores) + 1e-8) + 1e-8)
            perturbed_scores = scores + gumbel_noise
            
            # Softmax com temperatura
            probs = F.softmax(perturbed_scores / temperature, dim=0)
            
            # Selecionar top-k
            _, topk_indices = torch.topk(probs, k)
            weights = probs[topk_indices]
        else:
            # Durante inferência: seleção hard (determinística)
            _, topk_indices = torch.topk(scores, k)
            weights = torch.ones(k, device=scores.device)
        
        return topk_indices, weights
    
    def _filter_edges(self, edge_index, selected_indices):
        """
        Filtra arestas para manter apenas as que conectam vértices selecionados.
        
        Args:
            edge_index: Arestas originais (2, E)
            selected_indices: Índices dos vértices selecionados (K,)
            
        Returns:
            edge_index_filtered: Arestas filtradas (2, E')
        """
        if edge_index.shape[1] == 0:
            return edge_index
        
        # Criar conjunto de índices selecionados para lookup rápido
        selected_set = set(selected_indices.cpu().numpy().tolist())
        
        # Filtrar arestas onde ambos os vértices estão selecionados
        mask = []
        for i in range(edge_index.shape[1]):
            v0 = edge_index[0, i].item()
            v1 = edge_index[1, i].item()
            if v0 in selected_set and v1 in selected_set:
                mask.append(i)
        
        if len(mask) == 0:
            # Se não há arestas, criar grafo vazio
            return torch.empty((2, 0), dtype=torch.long, device=edge_index.device)
        
        filtered_edges = edge_index[:, mask]
        
        # Criar mapeamento: índice original → novo índice
        n_original = max(selected_indices.max().item() + 1, edge_index.max().item() + 1)
        mapping = torch.full((n_original,), -1, dtype=torch.long, device=edge_index.device)
        mapping[selected_indices] = torch.arange(len(selected_indices), device=edge_index.device)
        
        # Remapear índices
        filtered_edges[0] = mapping[filtered_edges[0]]
        filtered_edges[1] = mapping[filtered_edges[1]]
        
        return filtered_edges


# Manter compatibilidade: alias para o modelo antigo
RetopologyGNN = StableRetopologyGNN

