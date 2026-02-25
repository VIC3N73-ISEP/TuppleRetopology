"""
Dataset para ModelRetopology2000: pares (heatmap PLY, highpoly, lowpoly).
O heatmap PLY contém as cores por vértice = importância (vermelho = importante).
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
import trimesh
from pathlib import Path
from typing import Optional
import json

from config import MODEL_CONFIG, MESH_CONFIG, ORGANIZED_PAIRS_DIR


def _importance_from_heatmap_ply(heatmap_path: Path, num_vertices: int) -> np.ndarray:
    """
    Extrai o score de importância do PLY heatmap (vertex colors).
    Colormap usado: R = score*255, logo importance = R/255.
    Retorna array (num_vertices,) em [0, 1]. Se ficheiro não existir ou tamanho diferente, retorna 0.5.
    """
    if not heatmap_path.exists():
        return np.full(num_vertices, 0.5, dtype=np.float32)
    try:
        mesh = trimesh.load(str(heatmap_path))
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.geometry[list(mesh.geometry.keys())[0]]
        if len(mesh.vertices) != num_vertices:
            return np.full(num_vertices, 0.5, dtype=np.float32)
        if not hasattr(mesh.visual, "vertex_colors") or mesh.visual.vertex_colors is None:
            return np.full(num_vertices, 0.5, dtype=np.float32)
        colors = np.asarray(mesh.visual.vertex_colors[:, :3], dtype=np.float32)
        # Vermelho = importante: score = R/255
        importance = (colors[:, 0] / 255.0).astype(np.float32)
        return np.clip(importance, 0.0, 1.0)
    except Exception:
        return np.full(num_vertices, 0.5, dtype=np.float32)


class RetopologyDataset2000(Dataset):
    """
    Dataset para pares (heatmap PLY, highpoly, lowpoly).
    Cada amostra: highpoly grafo com 10 features (9 + importância do heatmap), lowpoly grafo (9 features, pad 10).
    """

    def __init__(self, pairs_dir, pairs_list, config=None):
        self.pairs_dir = Path(pairs_dir)
        self.pairs_list = pairs_list
        self.config = config or MODEL_CONFIG
        self.mesh_config = MESH_CONFIG
        metadata_file = self.pairs_dir.parent / "pairs_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, "r") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = None

    def __len__(self):
        return len(self.pairs_list)

    def mesh_to_graph(self, mesh: trimesh.Trimesh, importance: Optional[np.ndarray] = None) -> Data:
        """Converte mesh em grafo. Se importance for dado, usa 10 features (9 + importância)."""
        if len(mesh.vertices) > self.mesh_config["max_vertices"]:
            mesh = mesh.simplify_quadric_decimation(self.mesh_config["max_vertices"])
        if self.mesh_config["normalize"]:
            mesh = mesh.copy()
            mesh.vertices = mesh.vertices - mesh.vertices.mean(axis=0)
            scale = np.max(np.abs(mesh.vertices))
            if scale > 0:
                mesh.vertices = mesh.vertices / scale

        vertices = mesh.vertices.astype(np.float32)
        faces = mesh.faces
        if mesh.vertex_normals is not None and len(mesh.vertex_normals) > 0:
            normals = mesh.vertex_normals.astype(np.float32)
        else:
            normals = None

        if self.mesh_config["add_features"]:
            curvature = self._compute_curvature(mesh)
            center = vertices.mean(axis=0)
            dist_to_center = np.linalg.norm(vertices - center, axis=1, keepdims=True)
            if normals is not None:
                base = np.concatenate([
                    vertices, normals,
                    curvature.reshape(-1, 1), dist_to_center,
                    np.zeros((len(vertices), 1)),
                ], axis=1)
            else:
                base = np.concatenate([
                    vertices, np.zeros((len(vertices), 3)),
                    curvature.reshape(-1, 1), dist_to_center,
                    np.zeros((len(vertices), 1)),
                ], axis=1)
        else:
            base = np.concatenate([vertices, normals if normals is not None else np.zeros((len(vertices), 3))], axis=1)

        n_feat = self.config["node_features"]
        if importance is not None:
            imp = np.asarray(importance, dtype=np.float32).reshape(-1, 1)
            if imp.shape[0] != len(vertices):
                imp = np.full((len(vertices), 1), 0.5, dtype=np.float32)
            node_features = np.concatenate([base[:, :9], imp], axis=1)
        else:
            # Sem heatmap: garantir 9 colunas + 1 zero (importância)
            if base.shape[1] >= 9:
                node_features = np.concatenate([base[:, :9], np.zeros((len(vertices), 1), dtype=np.float32)], axis=1)
            else:
                pad = np.zeros((len(vertices), 9 - base.shape[1]), dtype=np.float32)
                node_features = np.concatenate([base, pad, np.zeros((len(vertices), 1), dtype=np.float32)], axis=1)
        if node_features.shape[1] < n_feat:
            pad = np.zeros((len(vertices), n_feat - node_features.shape[1]), dtype=np.float32)
            node_features = np.concatenate([node_features, pad], axis=1)
        else:
            node_features = node_features[:, :n_feat]

        edges = []
        for face in faces:
            edges.append([face[0], face[1]])
            edges.append([face[1], face[2]])
            edges.append([face[2], face[0]])
        if len(edges) == 0:
            num_vertices = len(vertices)
            max_n = min(6, num_vertices - 1)
            for i in range(num_vertices):
                dists = np.linalg.norm(vertices - vertices[i], axis=1)
                idx = np.argsort(dists)[1 : max_n + 1]
                for j in idx:
                    edges.append([i, j])
        edges = np.array(edges, dtype=np.int64).T
        edges_set = set()
        edges_clean = []
        for i in range(edges.shape[1]):
            e = tuple(sorted([edges[0, i], edges[1, i]]))
            if e not in edges_set:
                edges_set.add(e)
                edges_clean.append([e[0], e[1]])
                edges_clean.append([e[1], e[0]])
        if not edges_clean:
            edges_clean = [[0, 1], [1, 0]] if len(vertices) > 1 else []
        edges_clean = np.array(edges_clean, dtype=np.int64).T

        return Data(
            x=torch.from_numpy(node_features).float(),
            edge_index=torch.from_numpy(edges_clean).long(),
            pos=torch.from_numpy(vertices).float(),
        )

    def _compute_curvature(self, mesh: trimesh.Trimesh) -> np.ndarray:
        vertices = mesh.vertices
        curvature = np.zeros(len(vertices))
        if mesh.vertex_normals is not None and len(mesh.vertex_normals) > 0:
            normals = mesh.vertex_normals
            for i, vertex in enumerate(vertices):
                dists = np.linalg.norm(vertices - vertex, axis=1)
                neighbors = np.where((dists < np.percentile(dists, 10)) & (dists > 0))[0]
                if len(neighbors) > 0:
                    normal_var = np.var(normals[neighbors], axis=0)
                    curvature[i] = np.linalg.norm(normal_var)
        else:
            for i, vertex in enumerate(vertices):
                dists = np.linalg.norm(vertices - vertex, axis=1)
                neighbors = np.where((dists < np.percentile(dists, 10)) & (dists > 0))[0]
                if len(neighbors) > 0:
                    curvature[i] = np.mean(dists[neighbors])
        if curvature.max() > 0:
            curvature = curvature / curvature.max()
        return curvature.astype(np.float32)

    def __getitem__(self, idx):
        pair_id = self.pairs_list[idx]
        pair_dir = self.pairs_dir / f"pair_{pair_id}"
        highpoly_path = pair_dir / f"highpoly-{pair_id}.obj"
        lowpoly_path = pair_dir / f"lowpoly-{pair_id}.obj"
        heatmap_path = pair_dir / f"highpoly-{pair_id}_heatmap.ply"

        try:
            highpoly_mesh = trimesh.load(str(highpoly_path))
            lowpoly_mesh = trimesh.load(str(lowpoly_path))
            if isinstance(highpoly_mesh, trimesh.Scene):
                highpoly_mesh = highpoly_mesh.geometry[list(highpoly_mesh.geometry.keys())[0]]
            if isinstance(lowpoly_mesh, trimesh.Scene):
                lowpoly_mesh = lowpoly_mesh.geometry[list(lowpoly_mesh.geometry.keys())[0]]

            importance = _importance_from_heatmap_ply(heatmap_path, len(highpoly_mesh.vertices))
            highpoly_graph = self.mesh_to_graph(highpoly_mesh, importance=importance)
            lowpoly_graph = self.mesh_to_graph(lowpoly_mesh, importance=None)
            if lowpoly_graph.x.shape[1] < self.config["node_features"]:
                pad = torch.zeros((lowpoly_graph.x.shape[0], self.config["node_features"] - lowpoly_graph.x.shape[1]))
                lowpoly_graph.x = torch.cat([lowpoly_graph.x, pad], dim=1)

            return {"highpoly": highpoly_graph, "lowpoly": lowpoly_graph, "pair_id": pair_id}
        except Exception as e:
            print(f"Erro ao carregar par {pair_id}: {e}")
            empty = Data(
                x=torch.zeros((1, self.config["node_features"])),
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                pos=torch.zeros((1, 3)),
            )
            return {"highpoly": empty, "lowpoly": empty, "pair_id": pair_id}


def collate_fn(batch):
    from torch_geometric.data import Batch
    highpoly_batch = Batch.from_data_list([b["highpoly"] for b in batch])
    lowpoly_batch = Batch.from_data_list([b["lowpoly"] for b in batch])
    return {"highpoly": highpoly_batch, "lowpoly": lowpoly_batch, "pair_ids": [b["pair_id"] for b in batch]}


def get_data_loaders(train_pairs, val_pairs, batch_size=4):
    train_dataset = RetopologyDataset2000(ORGANIZED_PAIRS_DIR, train_pairs)
    val_dataset = RetopologyDataset2000(ORGANIZED_PAIRS_DIR, val_pairs)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)
    return train_loader, val_loader
