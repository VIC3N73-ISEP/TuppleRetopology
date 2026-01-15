"""
Dataset loader para pares de meshes high poly e low poly.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
import trimesh
from pathlib import Path
from typing import Tuple, Optional
import json

from config import MODEL_CONFIG, MESH_CONFIG, ORGANIZED_PAIRS_DIR


class RetopologyDataset(Dataset):
    """
    Dataset para pares de meshes high poly e low poly.
    """
    
    def __init__(self, pairs_dir, pairs_list, config=None):
        """
        Args:
            pairs_dir: Diretório com os pares organizados
            pairs_list: Lista de IDs dos pares a usar
            config: Configuração do modelo
        """
        self.pairs_dir = Path(pairs_dir)
        self.pairs_list = pairs_list
        self.config = config or MODEL_CONFIG
        self.mesh_config = MESH_CONFIG
        
        # Carregar metadados se existir
        metadata_file = self.pairs_dir.parent / "pairs_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = None
    
    def __len__(self):
        return len(self.pairs_list)
    
    def mesh_to_graph(self, mesh: trimesh.Trimesh) -> Data:
        """
        Converte um mesh trimesh em um grafo PyTorch Geometric.
        
        Args:
            mesh: Mesh do trimesh
            
        Returns:
            Data object do PyTorch Geometric
        """
        # Limitar número de vértices se necessário
        if len(mesh.vertices) > self.mesh_config["max_vertices"]:
            # Simplificar mesh
            mesh = mesh.simplify_quadric_decimation(self.mesh_config["max_vertices"])
        
        # Normalizar mesh se configurado
        if self.mesh_config["normalize"]:
            mesh.vertices = mesh.vertices - mesh.vertices.mean(axis=0)
            scale = np.max(np.abs(mesh.vertices))
            if scale > 0:
                mesh.vertices = mesh.vertices / scale
        
        vertices = mesh.vertices.astype(np.float32)
        faces = mesh.faces
        
        # Calcular normais dos vértices
        if mesh.vertex_normals is not None and len(mesh.vertex_normals) > 0:
            normals = mesh.vertex_normals.astype(np.float32)
        else:
            normals = mesh.vertex_normals
        
        # Features adicionais
        if self.mesh_config["add_features"]:
            # Curvatura (aproximada)
            curvature = self._compute_curvature(mesh)
            
            # Distância ao centro
            center = vertices.mean(axis=0)
            dist_to_center = np.linalg.norm(vertices - center, axis=1, keepdims=True)
            
            # Features combinadas
            if normals is not None and len(normals) > 0:
                node_features = np.concatenate([
                    vertices,  # 3D position
                    normals,   # 3D normal
                    curvature.reshape(-1, 1),  # curvature
                    dist_to_center,  # distance to center
                    np.zeros((len(vertices), 1))  # placeholder
                ], axis=1)
            else:
                # Se não houver normais, usar zeros
                node_features = np.concatenate([
                    vertices,
                    np.zeros((len(vertices), 3)),
                    curvature.reshape(-1, 1),
                    dist_to_center,
                    np.zeros((len(vertices), 1))
                ], axis=1)
        else:
            # Features básicas: posição + normal
            if normals is not None and len(normals) > 0:
                node_features = np.concatenate([vertices, normals], axis=1)
            else:
                node_features = np.concatenate([
                    vertices,
                    np.zeros((len(vertices), 3))
                ], axis=1)
        
        # Garantir que temos 9 features (pad ou trim)
        if node_features.shape[1] < self.config["node_features"]:
            padding = np.zeros((len(vertices), self.config["node_features"] - node_features.shape[1]))
            node_features = np.concatenate([node_features, padding], axis=1)
        elif node_features.shape[1] > self.config["node_features"]:
            node_features = node_features[:, :self.config["node_features"]]
        
        # Criar arestas a partir das faces
        edges = []
        for face in faces:
            # Cada face cria 3 arestas (não direcionadas)
            edges.append([face[0], face[1]])
            edges.append([face[1], face[2]])
            edges.append([face[2], face[0]])
        
        if len(edges) == 0:
            # Se não houver faces, criar arestas baseadas em proximidade
            # Calcular distâncias entre todos os pares
            num_vertices = len(vertices)
            max_neighbors = min(6, num_vertices - 1)
            
            for i in range(num_vertices):
                # Calcular distâncias deste vértice para todos os outros
                distances = np.linalg.norm(vertices - vertices[i], axis=1)
                # Pegar os k mais próximos (excluindo o próprio vértice)
                nearest_indices = np.argsort(distances)[1:max_neighbors+1]
                for j in nearest_indices:
                    edges.append([i, j])
        
        edges = np.array(edges, dtype=np.int64).T
        
        # Remover duplicatas e criar grafo não direcionado
        edges_set = set()
        edges_clean = []
        for i in range(edges.shape[1]):
            edge = tuple(sorted([edges[0, i], edges[1, i]]))
            if edge not in edges_set:
                edges_set.add(edge)
                edges_clean.append([edge[0], edge[1]])
                edges_clean.append([edge[1], edge[0]])  # Grafo não direcionado
        
        if len(edges_clean) == 0:
            # Fallback: criar arestas mínimas
            edges_clean = [[0, 1], [1, 0]] if len(vertices) > 1 else []
        
        edges_clean = np.array(edges_clean, dtype=np.int64).T
        
        # Converter para tensores
        x = torch.from_numpy(node_features).float()
        edge_index = torch.from_numpy(edges_clean).long()
        pos = torch.from_numpy(vertices).float()
        
        return Data(x=x, edge_index=edge_index, pos=pos)
    
    def _compute_curvature(self, mesh: trimesh.Trimesh) -> np.ndarray:
        """
        Calcula uma aproximação da curvatura para cada vértice.
        """
        vertices = mesh.vertices
        curvature = np.zeros(len(vertices))
        
        # Aproximação simples: variação das normais
        if mesh.vertex_normals is not None and len(mesh.vertex_normals) > 0:
            normals = mesh.vertex_normals
            # Calcular variação das normais dos vizinhos
            for i, vertex in enumerate(vertices):
                # Encontrar vértices próximos
                distances = np.linalg.norm(vertices - vertex, axis=1)
                neighbors = np.where((distances < np.percentile(distances, 10)) & (distances > 0))[0]
                
                if len(neighbors) > 0:
                    # Variação das normais
                    normal_var = np.var(normals[neighbors], axis=0)
                    curvature[i] = np.linalg.norm(normal_var)
        else:
            # Fallback: usar distância média aos vizinhos
            for i, vertex in enumerate(vertices):
                distances = np.linalg.norm(vertices - vertex, axis=1)
                neighbors = np.where((distances < np.percentile(distances, 10)) & (distances > 0))[0]
                if len(neighbors) > 0:
                    curvature[i] = np.mean(distances[neighbors])
        
        # Normalizar
        if curvature.max() > 0:
            curvature = curvature / curvature.max()
        
        return curvature
    
    def __getitem__(self, idx):
        """
        Retorna um par (high_poly, low_poly) de grafos.
        """
        pair_id = self.pairs_list[idx]
        pair_dir = self.pairs_dir / f"pair_{pair_id}"
        
        # Caminhos dos arquivos
        highpoly_path = pair_dir / f"highpoly-{pair_id}.obj"
        lowpoly_path = pair_dir / f"lowpoly-{pair_id}.obj"
        
        # Carregar meshes
        try:
            highpoly_mesh = trimesh.load(str(highpoly_path))
            lowpoly_mesh = trimesh.load(str(lowpoly_path))
            
            # Se for uma Scene, pegar o primeiro mesh
            if isinstance(highpoly_mesh, trimesh.Scene):
                highpoly_mesh = highpoly_mesh.geometry[list(highpoly_mesh.geometry.keys())[0]]
            if isinstance(lowpoly_mesh, trimesh.Scene):
                lowpoly_mesh = lowpoly_mesh.geometry[list(lowpoly_mesh.geometry.keys())[0]]
            
            # Converter para grafos
            highpoly_graph = self.mesh_to_graph(highpoly_mesh)
            lowpoly_graph = self.mesh_to_graph(lowpoly_mesh)
            
            return {
                "highpoly": highpoly_graph,
                "lowpoly": lowpoly_graph,
                "pair_id": pair_id
            }
            
        except Exception as e:
            print(f"Erro ao carregar par {pair_id}: {e}")
            # Retornar dados vazios em caso de erro
            empty_graph = Data(
                x=torch.zeros((1, self.config["node_features"])),
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                pos=torch.zeros((1, 3))
            )
            return {
                "highpoly": empty_graph,
                "lowpoly": empty_graph,
                "pair_id": pair_id
            }


def collate_fn(batch):
    """
    Função de collate para o DataLoader.
    """
    from torch_geometric.data import Batch
    
    highpoly_batch = Batch.from_data_list([item["highpoly"] for item in batch])
    lowpoly_batch = Batch.from_data_list([item["lowpoly"] for item in batch])
    pair_ids = [item["pair_id"] for item in batch]
    
    return {
        "highpoly": highpoly_batch,
        "lowpoly": lowpoly_batch,
        "pair_ids": pair_ids
    }


def get_data_loaders(train_pairs, val_pairs, batch_size=4):
    """
    Cria os data loaders para treino e validação.
    
    Args:
        train_pairs: Lista de IDs dos pares para treino
        val_pairs: Lista de IDs dos pares para validação
        batch_size: Tamanho do batch
        
    Returns:
        train_loader, val_loader
    """
    train_dataset = RetopologyDataset(ORGANIZED_PAIRS_DIR, train_pairs)
    val_dataset = RetopologyDataset(ORGANIZED_PAIRS_DIR, val_pairs)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0  # Pode aumentar se necessário
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    return train_loader, val_loader

