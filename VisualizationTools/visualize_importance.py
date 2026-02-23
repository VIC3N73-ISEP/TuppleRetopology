"""
Ferramenta de visualização: heatmap de importância de vértices na malha.
Azul = pouco importante, Vermelho = muito importante.
Usa trimesh para vertex colors e viewer/export.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import trimesh

# Incluir ModuleVerticeRanker1000 no path para importar ranker (que usa config/metrics na mesma pasta)
_MODULE_RANKER_DIR = Path(__file__).resolve().parent.parent / "ModuleVerticeRanker1000"
if str(_MODULE_RANKER_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULE_RANKER_DIR))

from ranker import compute_importance_ranking


def load_mesh(path: Path) -> trimesh.Trimesh:
    """Carrega uma mesh; se for Scene, devolve o primeiro geometry."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Ficheiro não encontrado: {path}")
    mesh = trimesh.load(str(path))
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.geometry[list(mesh.geometry.keys())[0]]
    if not hasattr(mesh, "vertices") or len(mesh.vertices) == 0:
        raise ValueError(f"Malha inválida ou vazia: {path}")
    return mesh


def load_scores_from_file(scores_path: Path, num_vertices: int) -> np.ndarray:
    """
    Carrega scores de um ficheiro JSON ou CSV.
    Retorna array (num_vertices,) na ordem do índice do vértice.
    """
    scores_path = Path(scores_path)
    if not scores_path.exists():
        raise FileNotFoundError(f"Ficheiro de scores não encontrado: {scores_path}")

    suffix = scores_path.suffix.lower()
    if suffix == ".json":
        with open(scores_path, encoding="utf-8") as f:
            data = json.load(f)
        scores = np.array(data["scores"], dtype=np.float64)
    elif suffix == ".csv":
        scores = np.full(num_vertices, np.nan)
        with open(scores_path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if "vertex_index" not in reader.fieldnames or "score" not in reader.fieldnames:
                raise ValueError("CSV deve ter colunas 'vertex_index' e 'score'")
            for row in reader:
                vi = int(row["vertex_index"])
                s = float(row["score"])
                if vi < 0 or vi >= num_vertices:
                    raise ValueError(
                        f"vertex_index {vi} fora do intervalo [0, {num_vertices - 1}]"
                    )
                scores[vi] = s
        if np.any(np.isnan(scores)):
            raise ValueError("CSV não cobre todos os vértices (vertex_index 0..N-1)")
    else:
        raise ValueError("Formato de scores não suportado; use .json ou .csv")

    if len(scores) != num_vertices:
        raise ValueError(
            f"Número de scores ({len(scores)}) não coincide com número de vértices ({num_vertices})"
        )
    return scores


def scores_to_colors(scores: np.ndarray) -> np.ndarray:
    """
    Mapeia scores em [0, 1] para cores RGBA: azul (0) -> vermelho (1).
    Retorna array (N, 4) uint8.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if scores.size == 0:
        return np.zeros((0, 4), dtype=np.uint8)
    s = np.clip(scores, 0.0, 1.0)
    # Linear: R = score, B = 1 - score, G suave no meio
    r = (s * 255).astype(np.uint8)
    b = ((1.0 - s) * 255).astype(np.uint8)
    g = (np.abs(2.0 * s - 1.0) * 0.4 * 255).astype(np.uint8)  # mais escuro nos extremos
    alpha = np.full_like(r, 255)
    return np.stack([r, g, b, alpha], axis=1)


def apply_vertex_colors(mesh: trimesh.Trimesh, colors: np.ndarray) -> None:
    """Atribui cores por vértice à mesh (in-place). colors: (N, 4) uint8 RGBA."""
    if len(colors) != len(mesh.vertices):
        raise ValueError(
            f"Número de cores ({len(colors)}) não coincide com número de vértices ({len(mesh.vertices)})"
        )
    if mesh.visual is None or not hasattr(mesh.visual, "vertex_colors"):
        mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh)
    mesh.visual.vertex_colors = np.asarray(colors, dtype=np.uint8)


def main():
    parser = argparse.ArgumentParser(
        description="Visualização: heatmap de importância de vértices (azul → vermelho)."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Malha de input (OBJ, PLY, STL, etc.)",
    )
    parser.add_argument(
        "--scores",
        "-s",
        type=Path,
        default=None,
        help="Ficheiro de scores JSON ou CSV (opcional; se omitido, calcula com o ranker)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Guardar malha colorida (ex.: .ply ou .obj)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Abrir viewer trimesh após colorir",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Não abrir viewer (útil quando só se quer --output)",
    )
    args = parser.parse_args()

    try:
        mesh = load_mesh(args.input)
    except (FileNotFoundError, ValueError) as e:
        print(f"Erro: {e}", file=sys.stderr)
        sys.exit(1)

    num_vertices = len(mesh.vertices)

    if args.scores is not None:
        try:
            scores = load_scores_from_file(args.scores, num_vertices)
        except (FileNotFoundError, ValueError) as e:
            print(f"Erro ao carregar scores: {e}", file=sys.stderr)
            sys.exit(1)
        # Normalizar para [0, 1] se necessário
        if scores.max() > scores.min():
            scores = (scores - scores.min()) / (scores.max() - scores.min())
        else:
            scores = np.zeros_like(scores)
    else:
        scores, _, _ = compute_importance_ranking(mesh)
        # scores já vêm em [0, 1]

    colors = scores_to_colors(scores)
    apply_vertex_colors(mesh, colors)

    if args.output is not None:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(str(out_path))
        print(f"Malha colorida guardada: {out_path}")

    do_show = args.show or (not args.no_show and args.output is None)
    if do_show:
        try:
            mesh.show()
        except (ModuleNotFoundError, ImportError) as e:
            msg = str(e).lower()
            name = getattr(e, "name", "") or ""
            if name == "pyglet" or "pyglet" in msg:
                if "pyglet<2" in msg or "pyglet<2" in str(e):
                    print(
                        "Viewer trimesh exige pyglet versão 1.x (a 2.x não é compatível).\n"
                        '  Opção 1: instale com  pip install "pyglet<2"\n'
                        "  Opção 2: guarde a malha com  --output ficheiro.ply  e abra noutro programa.",
                        file=sys.stderr,
                    )
                else:
                    print(
                        "Viewer não disponível: falta o módulo 'pyglet'.\n"
                        "  Opção 1: instale com  pip install \"pyglet<2\"\n"
                        "  Opção 2: guarde a malha com  --output ficheiro.ply  e abra noutro programa.",
                        file=sys.stderr,
                    )
            else:
                raise

    if not do_show and args.output is None:
        print("Nada a mostrar. Use --show para abrir o viewer ou --output para guardar.")


if __name__ == "__main__":
    main()
