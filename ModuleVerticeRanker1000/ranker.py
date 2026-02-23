"""
Orquestração do ranking de importância: chama as 3 métricas, normaliza, combina e ordena.
"""

import numpy as np
from typing import Tuple, Dict, Optional

import trimesh

from config import VERTICE_RANKER_WEIGHTS, MESH_CONFIG
from metrics import quadric_importance, curvature_importance, normal_deviation_importance


def compute_importance_ranking(
    mesh: trimesh.Trimesh,
    weights: Optional[Dict[str, float]] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Calcula o ranking de importância dos vértices da mesh.
    Não depende do número de vértices: usa a malha introduzida tal como está
    e rankeia todos os seus vértices, do mais ao menos importante.

    Args:
        mesh: Mesh trimesh (vértices, faces, vertex_normals).
        weights: Pesos para quadric, curvature, normal_deviation. Se None, usa config.

    Returns:
        scores: Array (V,) com score final por vértice (maior = mais importante).
        ranking: Array (V,) com índices dos vértices ordenados por importância decrescente.
        details: Dict com as métricas normalizadas por vértice:
            "quadric", "curvature", "normal_deviation".
    """
    weights = weights or VERTICE_RANKER_WEIGHTS
    w_q = weights.get("quadric", 1.0 / 3.0)
    w_c = weights.get("curvature", 1.0 / 3.0)
    w_d = weights.get("normal_deviation", 1.0 / 3.0)

    # Usar a mesh tal como foi introduzida: ranking para todos os vértices (sem limite de número).
    # Opcional: apenas normalizar escala/centro.
    if MESH_CONFIG.get("normalize"):
        mesh = mesh.copy()
        mesh.vertices = mesh.vertices - mesh.vertices.mean(axis=0)
        scale = np.max(np.abs(mesh.vertices))
        if scale > 0:
            mesh.vertices = mesh.vertices / scale

    quadric = quadric_importance(mesh)
    curvature = curvature_importance(mesh)
    normal_dev = normal_deviation_importance(mesh)

    # Já vêm em [0, 1]; combinar com pesos
    scores = w_q * quadric + w_c * curvature + w_d * normal_dev
    # Normalizar scores para [0, 1]
    if scores.max() > scores.min():
        scores = (scores - scores.min()) / (scores.max() - scores.min())
    else:
        scores = np.zeros_like(scores)

    ranking = np.argsort(-scores)  # decrescente

    details = {
        "quadric": quadric,
        "curvature": curvature,
        "normal_deviation": normal_dev,
    }
    return scores, ranking, details
