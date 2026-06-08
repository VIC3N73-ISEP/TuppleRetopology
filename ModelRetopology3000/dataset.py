"""
Dataset ModelRetopology3000 (ratio-only): par (high, low) e r_gt derivado das faces.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
import trimesh
from pathlib import Path
import json

from config import (
    MODEL_CONFIG,
    MESH_CONFIG,
    ORGANIZED_PAIRS_DIR,
    ratio_to_target_face_count,
    r_gt_from_face_counts,
)


def decimate_by_ratio(mesh: trimesh.Trimesh, r: float) -> trimesh.Trimesh:
    """
    Quadric decimate: r alto = mais redução.
    Compatível com versões de trimesh que esperam:
    - target_reduction em [0, 1], ou
    - número alvo de faces.
    Retorna cópia se não conseguir simplificar.
    """
    mesh = mesh.copy()
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        return mesh
    n_faces = len(mesh.faces)
    reduction = float(max(0.0, min(1.0, r)))
    target = ratio_to_target_face_count(n_faces, reduction)
    if reduction <= 0.0 or target >= n_faces:
        return mesh

    try:
        # API moderna: simplify_quadric_decimation(target_reduction=float)
        out = mesh.simplify_quadric_decimation(reduction)
    except Exception:
        try:
            # API antiga (algumas versões): simplify_quadric_decimation(target_faces=int)
            out = mesh.simplify_quadric_decimation(target)
        except Exception:
            out = None

    if isinstance(out, trimesh.Trimesh) and len(out.vertices) > 0 and len(out.faces) > 0:
        return out

    # Fallback final: manter original se simplificação falhar ou vier inválida.
    return mesh


class RetopologyDataset3000(Dataset):
    """
    Por amostra: grafo high (9 feat) + escalar r_gt.
    """

    def __init__(self, pairs_dir, pairs_list, config=None):
        self.pairs_dir = Path(pairs_dir)
        self.pairs_list = pairs_list
        self.config = config or MODEL_CONFIG
        self.mesh_config = MESH_CONFIG
        self.base_f = self.config["base_node_features"]
        self.n_feat = self.config["node_features"]
        metadata_file = self.pairs_dir.parent / "pairs_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, "r") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = None

    def __len__(self):
        return len(self.pairs_list)

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

    def _normalize_mesh(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        m = mesh.copy()
        if self.mesh_config["normalize"]:
            m.vertices = m.vertices - m.vertices.mean(axis=0)
            scale = np.max(np.abs(m.vertices))
            if scale > 0:
                m.vertices = m.vertices / scale
        return m

    def _mesh_to_features_and_edges(self, mesh: trimesh.Trimesh):
        """Vértices + node_features base (9 colunas) + arestas."""
        if len(mesh.vertices) > self.mesh_config["max_vertices"]:
            mesh = mesh.simplify_quadric_decimation(self.mesh_config["max_vertices"])
        mesh = self._normalize_mesh(mesh)

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
                base = np.concatenate(
                    [
                        vertices,
                        normals,
                        curvature.reshape(-1, 1),
                        dist_to_center,
                        np.zeros((len(vertices), 1)),
                    ],
                    axis=1,
                )
            else:
                base = np.concatenate(
                    [
                        vertices,
                        np.zeros((len(vertices), 3)),
                        curvature.reshape(-1, 1),
                        dist_to_center,
                        np.zeros((len(vertices), 1)),
                    ],
                    axis=1,
                )
        else:
            base = np.concatenate(
                [vertices, normals if normals is not None else np.zeros((len(vertices), 3))],
                axis=1,
            )

        node_features = base[:, : self.base_f]
        if node_features.shape[1] < self.base_f:
            pad = np.zeros((len(vertices), self.base_f - node_features.shape[1]), dtype=np.float32)
            node_features = np.concatenate([node_features, pad], axis=1)

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

        return (
            vertices,
            node_features.astype(np.float32),
            edges_clean,
        )

    def mesh_to_graph_high(self, mesh: trimesh.Trimesh) -> Data:
        """Grafo high: só 9 features (entrada para RatioHead)."""
        vertices, nf, edges_clean = self._mesh_to_features_and_edges(mesh)
        x = torch.from_numpy(nf).float()
        edge_index = torch.from_numpy(edges_clean).long()
        pos = torch.from_numpy(vertices).float()
        return Data(x=x, edge_index=edge_index, pos=pos)

    def mesh_to_graph_refine(self, mesh: trimesh.Trimesh, r: float) -> Data:
        """Grafo decimado: 9 + ratio por vértice."""
        vertices, nf, edges_clean = self._mesh_to_features_and_edges(mesh)
        r_col = np.full((len(vertices), 1), float(r), dtype=np.float32)
        full = np.concatenate([nf, r_col], axis=1)
        if full.shape[1] < self.n_feat:
            pad = np.zeros((len(vertices), self.n_feat - full.shape[1]), dtype=np.float32)
            full = np.concatenate([full, pad], axis=1)
        else:
            full = full[:, : self.n_feat]
        return Data(
            x=torch.from_numpy(full).float(),
            edge_index=torch.from_numpy(edges_clean).long(),
            pos=torch.from_numpy(vertices).float(),
        )

    def __getitem__(self, idx):
        pair_id = self.pairs_list[idx]
        pair_dir = self.pairs_dir / f"pair_{pair_id}"
        highpoly_path = pair_dir / f"highpoly-{pair_id}.obj"
        lowpoly_path = pair_dir / f"lowpoly-{pair_id}.obj"

        try:
            highpoly_mesh = trimesh.load(str(highpoly_path))
            lowpoly_mesh = trimesh.load(str(lowpoly_path))
            if isinstance(highpoly_mesh, trimesh.Scene):
                highpoly_mesh = highpoly_mesh.geometry[list(highpoly_mesh.geometry.keys())[0]]
            if isinstance(lowpoly_mesh, trimesh.Scene):
                lowpoly_mesh = lowpoly_mesh.geometry[list(lowpoly_mesh.geometry.keys())[0]]

            n_f_high = len(highpoly_mesh.faces) if hasattr(highpoly_mesh, "faces") else 0
            n_f_low = len(lowpoly_mesh.faces) if hasattr(lowpoly_mesh, "faces") else 0
            if n_f_high < 1:
                n_f_high = 1
            n_f_low = max(0, n_f_low)

            r_gt = r_gt_from_face_counts(n_f_high, n_f_low)
            r_applied_gt = 1.0 - r_gt

            high_graph = self.mesh_to_graph_high(highpoly_mesh)

            return {
                "high": high_graph,
                "r_gt": torch.tensor(r_gt, dtype=torch.float32),
                "r_applied_gt": torch.tensor(r_applied_gt, dtype=torch.float32),
                "pair_id": pair_id,
            }
        except Exception as e:
            print(f"Erro ao carregar par {pair_id}: {e}")
            z = torch.zeros((1, self.base_f))
            empty = Data(
                x=z,
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                pos=torch.zeros((1, 3)),
            )
            return {
                "high": empty,
                "r_gt": torch.tensor(0.0, dtype=torch.float32),
                "r_applied_gt": torch.tensor(1.0, dtype=torch.float32),
                "pair_id": pair_id,
            }


def collate_fn(batch):
    from torch_geometric.data import Batch

    high_batch = Batch.from_data_list([b["high"] for b in batch])
    r_gt = torch.stack([b["r_gt"] for b in batch])
    r_applied_gt = torch.stack([b["r_applied_gt"] for b in batch])
    return {
        "high": high_batch,
        "r_gt": r_gt,
        "r_applied_gt": r_applied_gt,
        "pair_ids": [b["pair_id"] for b in batch],
    }


def get_data_loaders(train_pairs, val_pairs, batch_size=4):
    train_dataset = RetopologyDataset3000(ORGANIZED_PAIRS_DIR, train_pairs)
    val_dataset = RetopologyDataset3000(ORGANIZED_PAIRS_DIR, val_pairs)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )
    return train_loader, val_loader
