"""
Métricas de importância de vértices: Quadric Error, Curvature-Based, Normal Deviation.
"""

import numpy as np
from typing import List, Tuple

import trimesh


def _edges_from_faces(faces: np.ndarray) -> np.ndarray:
    """Lista de arestas únicas (cada aresta uma vez) a partir das faces. Shape (2, E)."""
    edges = set()
    for face in faces:
        a, b, c = face[0], face[1], face[2]
        edges.add((min(a, b), max(a, b)))
        edges.add((min(b, c), max(b, c)))
        edges.add((min(c, a), max(c, a)))
    if not edges:
        return np.zeros((2, 0), dtype=np.int64)
    return np.array(list(edges), dtype=np.int64).T


def _vertex_neighbors_from_edges(edges: np.ndarray, num_vertices: int) -> List[np.ndarray]:
    """Para cada vértice, array de índices dos vizinhos (1-ring). edges shape (2, E)."""
    neighbors: List[List[int]] = [[] for _ in range(num_vertices)]
    for i in range(edges.shape[1]):
        u, v = int(edges[0, i]), int(edges[1, i])
        if v not in neighbors[u]:
            neighbors[u].append(v)
        if u not in neighbors[v]:
            neighbors[v].append(u)
    return [np.array(n, dtype=np.int64) for n in neighbors]


def _vertex_edges_from_edges(edges: np.ndarray, num_vertices: int) -> List[List[Tuple[int, int]]]:
    """Para cada vértice, lista de (vizinho, edge_id) para arestas incidentes. edge_id = índice da aresta em edges."""
    vertex_edges: List[List[Tuple[int, int]]] = [[] for _ in range(num_vertices)]
    for e in range(edges.shape[1]):
        u, v = int(edges[0, e]), int(edges[1, e])
        vertex_edges[u].append((v, e))
        vertex_edges[v].append((u, e))
    return vertex_edges


def quadric_importance(mesh: trimesh.Trimesh) -> np.ndarray:
    """
    Importância por Quadric Error (QEM): por vértice, máximo custo de colapso das arestas
    incidentes. Maior custo = mais importante (remover perde mais detalhe).
    Retorna array (V,) normalizado em [0, 1].
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    nv = len(vertices)
    if nv == 0:
        return np.array([], dtype=np.float64)

    # Quadric por face: plano n·x + d = 0 => q = [nx, ny, nz, d], Q = qq^T
    # Pontos da face para obter normal e d
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    e1 = v1 - v0
    e2 = v2 - v0
    n = np.cross(e1, e2)
    area = np.linalg.norm(n, axis=1, keepdims=True)
    np.place(area, area == 0, 1.0)
    n = n / area
    d = -np.sum(n * v0, axis=1, keepdims=True)
    q = np.hstack([n, d])   # (F, 4)
    # Q_f = qq^T por face -> guardar como (F, 4, 4)
    Q_faces = np.einsum("fi,fj->fij", q, q)

    # Quadric por vértice: soma dos quadrics das faces incidentes
    Q_vertex = np.zeros((nv, 4, 4), dtype=np.float64)
    for f_idx, face in enumerate(faces):
        for v_idx in face:
            Q_vertex[v_idx] += Q_faces[f_idx]

    # Arestas únicas
    edges = _edges_from_faces(faces)
    if edges.shape[1] == 0:
        return np.zeros(nv, dtype=np.float64)

    # Custo de colapso por aresta: min_x x'^T (Q_u + Q_v) x' com x' = [x, y, z, 1]
    # Q escrito como [[A, b], [b^T, c]]. Ótimo: v = -A^{-1} b, custo = [v,1]^T Q [v,1]
    edge_costs = np.zeros(edges.shape[1], dtype=np.float64)
    for e in range(edges.shape[1]):
        u, v = edges[0, e], edges[1, e]
        Q_uv = Q_vertex[u] + Q_vertex[v]
        A = Q_uv[:3, :3]
        b = Q_uv[:3, 3]
        c = Q_uv[3, 3]
        try:
            det = np.linalg.det(A)
            if np.abs(det) > 1e-12:
                v_opt = -np.linalg.solve(A, b)
                x_h = np.array([v_opt[0], v_opt[1], v_opt[2], 1.0])
                cost = float(x_h @ Q_uv @ x_h)
            else:
                # Fallback: usar ponto médio
                mid = (vertices[u] + vertices[v]) / 2
                x_h = np.array([mid[0], mid[1], mid[2], 1.0])
                cost = float(x_h @ Q_uv @ x_h)
        except np.linalg.LinAlgError:
            mid = (vertices[u] + vertices[v]) / 2
            x_h = np.array([mid[0], mid[1], mid[2], 1.0])
            cost = float(x_h @ Q_uv @ x_h)
        edge_costs[e] = max(cost, 0.0)

    # Por vértice: máximo custo das arestas incidentes
    vertex_edges = _vertex_edges_from_edges(edges, nv)
    importance = np.zeros(nv, dtype=np.float64)
    for i in range(nv):
        if vertex_edges[i]:
            importance[i] = max(edge_costs[e_idx] for _, e_idx in vertex_edges[i])
        else:
            importance[i] = 0.0

    # Normalizar para [0, 1]
    m = importance.max()
    if m > 0:
        importance = importance / m
    return importance


def curvature_importance(mesh: trimesh.Trimesh) -> np.ndarray:
    """
    Importância por curvatura (1-ring): variação das normais dos vizinhos.
    Alta curvatura = mais importante. Retorna array (V,) em [0, 1].
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    nv = len(vertices)
    if nv == 0:
        return np.array([], dtype=np.float64)

    edges = _edges_from_faces(faces)
    neighbors = _vertex_neighbors_from_edges(edges, nv)

    if mesh.vertex_normals is not None and len(mesh.vertex_normals) > 0:
        normals = np.asarray(mesh.vertex_normals, dtype=np.float64)
        if normals.shape[0] != nv:
            normals = np.zeros((nv, 3), dtype=np.float64)
    else:
        normals = np.zeros((nv, 3), dtype=np.float64)

    curvature = np.zeros(nv, dtype=np.float64)
    for i in range(nv):
        nb = neighbors[i]
        if len(nb) > 0 and np.any(np.linalg.norm(normals, axis=1) > 0):
            normal_var = np.var(normals[nb], axis=0)
            curvature[i] = np.linalg.norm(normal_var)
        elif len(nb) > 0:
            # Fallback: distância média aos vizinhos
            curvature[i] = np.mean(np.linalg.norm(vertices[nb] - vertices[i], axis=1))

    if curvature.max() > 0:
        curvature = curvature / curvature.max()
    return curvature


def normal_deviation_importance(mesh: trimesh.Trimesh) -> np.ndarray:
    """
    Importância por desvio de normal: 1 - média(n_i · n_j) nos vizinhos 1-ring.
    Maior desvio = mais importante. Retorna array (V,) em [0, 1].
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    nv = len(vertices)
    if nv == 0:
        return np.array([], dtype=np.float64)

    edges = _edges_from_faces(faces)
    neighbors = _vertex_neighbors_from_edges(edges, nv)

    if mesh.vertex_normals is not None and len(mesh.vertex_normals) > 0:
        normals = np.asarray(mesh.vertex_normals, dtype=np.float64)
        if normals.shape[0] != nv:
            normals = np.zeros((nv, 3), dtype=np.float64)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        np.place(norms, norms == 0, 1.0)
        normals = normals / norms
    else:
        normals = np.zeros((nv, 3), dtype=np.float64)

    deviation = np.zeros(nv, dtype=np.float64)
    for i in range(nv):
        nb = neighbors[i]
        if len(nb) > 0:
            dots = np.dot(normals[nb], normals[i])
            dots = np.clip(dots, -1.0, 1.0)
            deviation[i] = 1.0 - float(np.mean(dots))
        else:
            deviation[i] = 0.0

    if deviation.max() > 0:
        deviation = deviation / deviation.max()
    return deviation
