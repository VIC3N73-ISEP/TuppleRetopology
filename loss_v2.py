"""
Loss function melhorada para retopologia com termos realistas.

Componentes:
1. Chamfer Distance (geometria)
2. Normal Consistency (orientação)
3. Edge Length Regularization (arestas uniformes)
4. Degenerate Face Penalty (evitar faces ruins)
5. Local Smoothness (suavidade local)
6. Vertex Count Consistency (estabilidade)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImprovedRetopologyLoss(nn.Module):
    """
    Loss function completa para retopologia com termos realistas.
    """
    
    def __init__(
        self,
        pos_weight=1.0,
        normal_weight=0.5,
        edge_length_weight=0.3,
        degenerate_weight=0.2,
        smoothness_weight=0.2,
        vertex_count_weight=0.1,
        normal_consistency_weight=0.3
    ):
        super(ImprovedRetopologyLoss, self).__init__()
        
        self.pos_weight = pos_weight
        self.normal_weight = normal_weight
        self.edge_length_weight = edge_length_weight
        self.degenerate_weight = degenerate_weight
        self.smoothness_weight = smoothness_weight
        self.vertex_count_weight = vertex_count_weight
        self.normal_consistency_weight = normal_consistency_weight
    
    def forward(
        self,
        pred_positions,
        pred_features,
        target_positions,
        target_features=None,
        pred_edge_index=None,
        target_edge_index=None,
        target_vertex_count=None
    ):
        """
        Calcula a loss combinada.
        
        Args:
            pred_positions: Posições preditas (N, 3)
            pred_features: Features preditas (N, 6)
            target_positions: Posições alvo (M, 3)
            target_features: Features alvo (M, 6) - opcional
            pred_edge_index: Arestas do grafo predito (2, E) - opcional
            target_edge_index: Arestas do grafo alvo (2, E') - opcional
            target_vertex_count: Número esperado de vértices - opcional
        """
        device = pred_positions.device
        loss_dict = {}
        
        # ========== 1. CHAMFER DISTANCE (Geometria) ==========
        # Verificar se há posições válidas
        if pred_positions.shape[0] == 0 or target_positions.shape[0] == 0:
            # Usar zeros_like para manter conexão com grafo computacional
            chamfer_loss = pred_positions.sum() * 0.0 if pred_positions.shape[0] > 0 else target_positions.sum() * 0.0
            hausdorff_loss = chamfer_loss
        else:
            chamfer_loss = self._chamfer_distance(pred_positions, target_positions)
            hausdorff_loss = self._hausdorff_distance(pred_positions, target_positions)
        
        loss_dict["chamfer_loss"] = chamfer_loss.item() if not torch.isnan(chamfer_loss) else 0.0
        loss_dict["hausdorff_loss"] = hausdorff_loss.item() if not torch.isnan(hausdorff_loss) else 0.0
        
        # Combinar geometria
        geometry_loss = 0.8 * chamfer_loss + 0.2 * hausdorff_loss
        
        # Verificar se geometry_loss é NaN
        if torch.isnan(geometry_loss) or torch.isinf(geometry_loss):
            # Manter conexão com grafo computacional
            geometry_loss = pred_positions.sum() * 0.0
        
        # ========== 3. NORMAL CONSISTENCY ==========
        # Inicializar com tensor conectado ao grafo
        normal_loss = pred_positions.sum() * 0.0
        if pred_edge_index is not None and pred_edge_index.shape[1] > 0:
            pred_normals = self._compute_vertex_normals(pred_positions, pred_edge_index)
            
            # Verificar se pred_normals é válido
            if not torch.isnan(pred_normals).any() and not torch.isinf(pred_normals).any():
                if target_edge_index is not None and target_edge_index.shape[1] > 0:
                    target_normals = self._compute_vertex_normals(target_positions, target_edge_index)
                    if not torch.isnan(target_normals).any() and not torch.isinf(target_normals).any():
                        normal_loss = self._chamfer_distance(pred_normals, target_normals)
                elif target_features is not None and target_features.shape[1] >= 3:
                    target_normals = F.normalize(target_features[:, :3], p=2, dim=1)
                    normal_loss = self._chamfer_distance(pred_normals, target_normals)
            
            # Verificar NaN
            if torch.isnan(normal_loss) or torch.isinf(normal_loss):
                normal_loss = pred_positions.sum() * 0.0
        
        loss_dict["normal_consistency_loss"] = normal_loss.item() if not torch.isnan(normal_loss) else 0.0
        
        # ========== 4. EDGE LENGTH REGULARIZATION ==========
        # Inicializar com tensor conectado ao grafo
        edge_length_loss = pred_positions.sum() * 0.0
        if pred_edge_index is not None and pred_edge_index.shape[1] > 0:
            edge_length_loss = self._edge_length_regularization(pred_positions, pred_edge_index)
            # Verificar NaN
            if torch.isnan(edge_length_loss) or torch.isinf(edge_length_loss):
                edge_length_loss = pred_positions.sum() * 0.0
        loss_dict["edge_length_loss"] = edge_length_loss.item() if not torch.isnan(edge_length_loss) else 0.0
        
        # ========== 5. DEGENERATE FACE PENALTY ==========
        # Inicializar com tensor conectado ao grafo
        degenerate_loss = pred_positions.sum() * 0.0
        if pred_edge_index is not None:
            degenerate_loss = self._degenerate_face_penalty(pred_positions, pred_edge_index)
            # Verificar NaN
            if torch.isnan(degenerate_loss) or torch.isinf(degenerate_loss):
                degenerate_loss = pred_positions.sum() * 0.0
        loss_dict["degenerate_loss"] = degenerate_loss.item() if not torch.isnan(degenerate_loss) else 0.0
        
        # ========== 6. LOCAL SMOOTHNESS ==========
        # Inicializar com tensor conectado ao grafo
        smoothness_loss = pred_positions.sum() * 0.0
        if pred_edge_index is not None and pred_edge_index.shape[1] > 0:
            smoothness_loss = self._local_smoothness(pred_positions, pred_edge_index)
            # Verificar NaN
            if torch.isnan(smoothness_loss) or torch.isinf(smoothness_loss):
                smoothness_loss = pred_positions.sum() * 0.0
        loss_dict["smoothness_loss"] = smoothness_loss.item() if not torch.isnan(smoothness_loss) else 0.0
        
        # ========== 7. VERTEX COUNT CONSISTENCY ==========
        # Inicializar com tensor conectado ao grafo
        vertex_count_loss = pred_positions.sum() * 0.0
        if target_vertex_count is not None:
            actual_count = pred_positions.shape[0]
            # Usar pred_positions para criar tensor com gradiente
            count_tensor = pred_positions.sum() * 0.0 + float(actual_count)
            target_tensor = pred_positions.sum() * 0.0 + float(target_vertex_count)
            vertex_count_loss = F.mse_loss(count_tensor.unsqueeze(0), target_tensor.unsqueeze(0))
            
            # Verificar NaN
            if torch.isnan(vertex_count_loss) or torch.isinf(vertex_count_loss):
                vertex_count_loss = pred_positions.sum() * 0.0
        loss_dict["vertex_count_loss"] = vertex_count_loss.item() if not torch.isnan(vertex_count_loss) else 0.0
        
        # ========== 8. FEATURE LOSS ==========
        # Inicializar com tensor conectado ao grafo
        feature_loss = pred_positions.sum() * 0.0
        if target_features is not None and pred_features is not None:
            if pred_features.shape[0] > 0 and target_features.shape[0] > 0:
                # Verificar se há valores NaN ou Inf
                if not (torch.isnan(pred_features).any() or torch.isinf(pred_features).any() or
                        torch.isnan(target_features).any() or torch.isinf(target_features).any()):
                    feature_distances = torch.cdist(pred_features, target_features)
                    min_feature_dist_pred = torch.min(feature_distances, dim=1)[0]
                    min_feature_dist_target = torch.min(feature_distances, dim=0)[0]
                    feature_loss = torch.mean(min_feature_dist_pred) + torch.mean(min_feature_dist_target)
                    
                    # Verificar se resultado é NaN
                    if torch.isnan(feature_loss) or torch.isinf(feature_loss):
                        feature_loss = pred_positions.sum() * 0.0
        loss_dict["feature_loss"] = feature_loss.item() if not torch.isnan(feature_loss) else 0.0
        
        # ========== LOSS TOTAL ==========
        # Garantir que todos os termos são válidos antes de somar
        safe_geometry_loss = geometry_loss if not (torch.isnan(geometry_loss) or torch.isinf(geometry_loss)) else pred_positions.sum() * 0.0
        safe_normal_loss = normal_loss if not (torch.isnan(normal_loss) or torch.isinf(normal_loss)) else pred_positions.sum() * 0.0
        safe_edge_length_loss = edge_length_loss if not (torch.isnan(edge_length_loss) or torch.isinf(edge_length_loss)) else pred_positions.sum() * 0.0
        safe_degenerate_loss = degenerate_loss if not (torch.isnan(degenerate_loss) or torch.isinf(degenerate_loss)) else pred_positions.sum() * 0.0
        safe_smoothness_loss = smoothness_loss if not (torch.isnan(smoothness_loss) or torch.isinf(smoothness_loss)) else pred_positions.sum() * 0.0
        safe_vertex_count_loss = vertex_count_loss if not (torch.isnan(vertex_count_loss) or torch.isinf(vertex_count_loss)) else pred_positions.sum() * 0.0
        safe_feature_loss = feature_loss if not (torch.isnan(feature_loss) or torch.isinf(feature_loss)) else pred_positions.sum() * 0.0
        
        total_loss = (
            self.pos_weight * safe_geometry_loss +
            self.normal_consistency_weight * safe_normal_loss +
            self.edge_length_weight * safe_edge_length_loss +
            self.degenerate_weight * safe_degenerate_loss +
            self.smoothness_weight * safe_smoothness_loss +
            self.vertex_count_weight * safe_vertex_count_loss +
            self.normal_weight * safe_feature_loss
        )
        
        # Verificar se total_loss é NaN ou Inf
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            # Se ainda for NaN, usar apenas geometry_loss (que sabemos que funciona)
            total_loss = safe_geometry_loss
        
        loss_dict["total_loss"] = total_loss.item() if not torch.isnan(total_loss) else 0.0
        
        return total_loss, loss_dict
    
    def _chamfer_distance(self, pred, target):
        """Chamfer Distance entre dois conjuntos de pontos."""
        if pred.shape[0] == 0 or target.shape[0] == 0:
            # Manter conexão com grafo computacional
            return pred.sum() * 0.0 if pred.shape[0] > 0 else target.sum() * 0.0
        
        # Verificar se há valores NaN ou Inf
        if torch.isnan(pred).any() or torch.isinf(pred).any():
            return pred.sum() * 0.0
        if torch.isnan(target).any() or torch.isinf(target).any():
            return target.sum() * 0.0
        
        distances = torch.cdist(pred, target)
        min_dist_pred = torch.min(distances, dim=1)[0]
        min_dist_target = torch.min(distances, dim=0)[0]
        
        result = torch.mean(min_dist_pred) + torch.mean(min_dist_target)
        
        # Verificar se resultado é NaN
        if torch.isnan(result) or torch.isinf(result):
            return pred.sum() * 0.0
        
        return result
    
    def _hausdorff_distance(self, pred, target):
        """Hausdorff Distance (distância máxima)."""
        if pred.shape[0] == 0 or target.shape[0] == 0:
            # Manter conexão com grafo computacional
            return pred.sum() * 0.0 if pred.shape[0] > 0 else target.sum() * 0.0
        
        # Verificar se há valores NaN ou Inf
        if torch.isnan(pred).any() or torch.isinf(pred).any():
            return pred.sum() * 0.0
        if torch.isnan(target).any() or torch.isinf(target).any():
            return target.sum() * 0.0
        
        distances = torch.cdist(pred, target)
        max_dist_pred_to_target = torch.max(torch.min(distances, dim=1)[0])
        max_dist_target_to_pred = torch.max(torch.min(distances, dim=0)[0])
        result = torch.max(max_dist_pred_to_target, max_dist_target_to_pred)
        
        # Verificar se resultado é NaN
        if torch.isnan(result) or torch.isinf(result):
            return pred.sum() * 0.0
        
        return result
    
    def _compute_vertex_normals(self, positions, edge_index):
        """Calcula normais dos vértices a partir da topologia."""
        if edge_index.shape[1] == 0 or positions.shape[0] == 0:
            return torch.zeros_like(positions)
        
        num_vertices = positions.shape[0]
        normals = torch.zeros_like(positions)
        
        # Verificar se os índices em edge_index são válidos
        max_idx = edge_index.max().item() if edge_index.numel() > 0 else -1
        if max_idx >= num_vertices:
            # Edge index contém índices inválidos, retornar zeros
            return normals
        
        for v in range(num_vertices):
            mask = (edge_index[0] == v) | (edge_index[1] == v)
            neighbor_edges = edge_index[:, mask]
            
            if neighbor_edges.shape[1] == 0:
                continue
            
            neighbors = []
            for i in range(neighbor_edges.shape[1]):
                if neighbor_edges[0, i] == v:
                    n_idx = neighbor_edges[1, i].item()
                else:
                    n_idx = neighbor_edges[0, i].item()
                
                # Verificar se o índice é válido
                if 0 <= n_idx < num_vertices:
                    neighbors.append(n_idx)
            
            neighbors = list(set(neighbors))
            
            if len(neighbors) < 2:
                continue
            
            vertex_normals = []
            center = positions[v]
            
            for i in range(len(neighbors)):
                n1_idx = neighbors[i]
                n2_idx = neighbors[(i + 1) % len(neighbors)]
                
                # Verificar índices novamente (segurança extra)
                if n1_idx >= num_vertices or n2_idx >= num_vertices:
                    continue
                
                vec1 = positions[n1_idx] - center
                vec2 = positions[n2_idx] - center
                
                normal = torch.linalg.cross(vec1, vec2)
                norm = torch.norm(normal)
                
                if norm > 1e-6:
                    normal = normal / norm
                    vertex_normals.append(normal)
            
            if len(vertex_normals) > 0:
                vertex_normals_tensor = torch.stack(vertex_normals)
                avg_normal = torch.mean(vertex_normals_tensor, dim=0)
                norm = torch.norm(avg_normal)
                if norm > 1e-6:
                    normals[v] = avg_normal / norm
        
        return normals
    
    def _edge_length_regularization(self, positions, edge_index):
        """
        Regulariza comprimentos de arestas para serem uniformes.
        Penaliza arestas muito longas ou muito curtas.
        """
        if edge_index.shape[1] == 0:
            return positions.sum() * 0.0
        
        # Verificar índices válidos
        max_idx = edge_index.max().item() if edge_index.numel() > 0 else -1
        if max_idx >= positions.shape[0]:
            return positions.sum() * 0.0
        
        # Calcular comprimentos das arestas
        edge_vectors = positions[edge_index[1]] - positions[edge_index[0]]
        edge_lengths = torch.norm(edge_vectors, dim=1)
        
        # Verificar se há NaN ou Inf
        if torch.isnan(edge_lengths).any() or torch.isinf(edge_lengths).any():
            return positions.sum() * 0.0
        
        # Penalizar variação (desvio padrão)
        if edge_lengths.shape[0] > 1:
            length_variance = torch.var(edge_lengths)
        else:
            length_variance = positions.sum() * 0.0
        
        # Penalizar arestas muito curtas (degeneradas)
        short_edge_penalty = torch.mean(F.relu(0.01 - edge_lengths))
        
        result = length_variance + 10.0 * short_edge_penalty
        
        # Verificar NaN
        if torch.isnan(result) or torch.isinf(result):
            return positions.sum() * 0.0
        
        return result
    
    def _degenerate_face_penalty(self, positions, edge_index):
        """
        Penaliza faces degeneradas (triângulos com área muito pequena).
        Usa arestas para estimar triângulos.
        """
        if edge_index.shape[1] < 3:
            return positions.sum() * 0.0
        
        # Para cada vértice, encontrar triângulos formados por suas arestas
        degenerate_penalty = positions.sum() * 0.0
        num_triangles = 0
        
        # Agrupar arestas por vértice
        vertex_edges = {}
        num_vertices = positions.shape[0]
        
        for i in range(edge_index.shape[1]):
            v0, v1 = edge_index[0, i].item(), edge_index[1, i].item()
            
            # Verificar índices válidos
            if v0 >= num_vertices or v1 >= num_vertices:
                continue
            
            if v0 not in vertex_edges:
                vertex_edges[v0] = []
            vertex_edges[v0].append(v1)
        
        # Para cada vértice, verificar triângulos
        for v, neighbors in vertex_edges.items():
            if v >= num_vertices or len(neighbors) < 2:
                continue
            
            # Verificar triângulos formados por pares de vizinhos
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    n1, n2 = neighbors[i], neighbors[j]
                    
                    # Verificar índices válidos
                    if n1 >= num_vertices or n2 >= num_vertices:
                        continue
                    
                    # Verificar se há aresta entre n1 e n2 (formando triângulo)
                    if n2 in vertex_edges.get(n1, []):
                        # Calcular área do triângulo
                        v_pos = positions[v]
                        n1_pos = positions[n1]
                        n2_pos = positions[n2]
                        
                        edge1 = n1_pos - v_pos
                        edge2 = n2_pos - v_pos
                        
                        # Área = 0.5 * ||cross(edge1, edge2)||
                        cross_product = torch.linalg.cross(edge1, edge2)
                        area = 0.5 * torch.norm(cross_product)
                        
                        # Verificar se área é válida
                        if torch.isnan(area) or torch.isinf(area):
                            continue
                        
                        # Penalizar áreas muito pequenas
                        if area < 1e-6:
                            penalty = (1e-6 - area) * 100.0
                            if not torch.isnan(penalty) and not torch.isinf(penalty):
                                degenerate_penalty = degenerate_penalty + penalty
                        
                        num_triangles += 1
        
        if num_triangles > 0:
            degenerate_penalty = degenerate_penalty / num_triangles
            # Verificar NaN
            if torch.isnan(degenerate_penalty) or torch.isinf(degenerate_penalty):
                return positions.sum() * 0.0
        
        return degenerate_penalty
    
    def _local_smoothness(self, positions, edge_index):
        """
        Penaliza variações bruscas na superfície (suavidade local).
        Mede a variação das normais entre vértices adjacentes.
        """
        if edge_index.shape[1] == 0:
            return positions.sum() * 0.0
        
        normals = self._compute_vertex_normals(positions, edge_index)
        
        # Para cada aresta, calcular diferença entre normais
        normal_diffs = []
        for i in range(edge_index.shape[1]):
            v0, v1 = edge_index[0, i].item(), edge_index[1, i].item()
            
            # Verificar índices válidos
            if v0 >= positions.shape[0] or v1 >= positions.shape[0]:
                continue
            
            n0, n1 = normals[v0], normals[v1]
            
            # Verificar se normais são válidas (não zero)
            n0_norm = torch.norm(n0)
            n1_norm = torch.norm(n1)
            
            if n0_norm < 1e-6 or n1_norm < 1e-6:
                continue
            
            # Normalizar
            n0_normalized = n0 / n0_norm
            n1_normalized = n1 / n1_norm
            
            # Diferença angular (dot product)
            dot = torch.clamp(torch.dot(n0_normalized, n1_normalized), -1.0 + 1e-7, 1.0 - 1e-7)
            angle_diff = torch.acos(dot)
            
            # Verificar se não é NaN
            if not torch.isnan(angle_diff) and not torch.isinf(angle_diff):
                normal_diffs.append(angle_diff)
        
        if len(normal_diffs) > 0:
            # Penalizar grandes variações
            smoothness = torch.mean(torch.stack(normal_diffs))
            # Verificar NaN
            if torch.isnan(smoothness) or torch.isinf(smoothness):
                return positions.sum() * 0.0
            return smoothness
        
        return positions.sum() * 0.0

