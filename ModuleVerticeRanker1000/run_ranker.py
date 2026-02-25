"""
CLI para executar o ranker de importância de vértices em ficheiros OBJ.
Com --input pasta: procura recursivamente OBJs com "highpoly" no nome,
gera heatmap e guarda o PLY na mesma pasta de cada OBJ.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import trimesh

from config import OUTPUT_DIR
from ranker import compute_importance_ranking


def scores_to_vertex_colors(scores: np.ndarray) -> np.ndarray:
    """Mapeia scores [0,1] para RGBA uint8: azul (0) -> vermelho (1). Shape (N, 4)."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.size == 0:
        return np.zeros((0, 4), dtype=np.uint8)
    s = np.clip(scores, 0.0, 1.0)
    r = (s * 255).astype(np.uint8)
    b = ((1.0 - s) * 255).astype(np.uint8)
    g = (np.abs(2.0 * s - 1.0) * 0.4 * 255).astype(np.uint8)
    alpha = np.full_like(r, 255)
    return np.stack([r, g, b, alpha], axis=1)


def export_heatmap_ply(mesh: trimesh.Trimesh, scores: np.ndarray, ply_path: Path) -> None:
    """Colore a mesh por scores e exporta para PLY."""
    colors = scores_to_vertex_colors(scores)
    if mesh.visual is None or not hasattr(mesh.visual, "vertex_colors"):
        mesh = mesh.copy()
        mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh)
    mesh.visual.vertex_colors = np.asarray(colors, dtype=np.uint8)
    mesh.export(str(ply_path))


def load_mesh(path: Path) -> Optional[trimesh.Trimesh]:
    """Carrega um mesh; se for Scene, devolve o primeiro geometry."""
    try:
        mesh = trimesh.load(str(path))
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.geometry[list(mesh.geometry.keys())[0]]
        return mesh
    except Exception as e:
        print(f"Erro ao carregar {path}: {e}", file=sys.stderr)
        return None


def run_on_mesh(
    mesh_path: Path,
    out_dir: Path,
    out_format: str,
    export_ply: bool = False,
) -> bool:
    """Executa o ranker numa mesh e guarda os resultados. Retorna True em sucesso."""
    mesh = load_mesh(mesh_path)
    if mesh is None:
        return False
    scores, ranking, details = compute_importance_ranking(mesh)
    n = len(scores)
    stem = mesh_path.stem

    if out_format == "json":
        out_path = out_dir / f"{stem}_ranking.json"
        data = {
            "mesh": str(mesh_path),
            "num_vertices": n,
            "ranking": ranking.tolist(),
            "scores": scores.tolist(),
            "quadric": details["quadric"].tolist(),
            "curvature": details["curvature"].tolist(),
            "normal_deviation": details["normal_deviation"].tolist(),
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"Guardado: {out_path}")
    else:
        out_path = out_dir / f"{stem}_ranking.csv"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("vertex_index,rank,score,quadric,curvature,normal_deviation\n")
            for r, idx in enumerate(ranking):
                f.write(
                    f"{idx},{r},{scores[idx]:.6f},"
                    f"{details['quadric'][idx]:.6f},"
                    f"{details['curvature'][idx]:.6f},"
                    f"{details['normal_deviation'][idx]:.6f}\n"
                )
        print(f"Guardado: {out_path}")

    if export_ply:
        ply_path = out_dir / f"{stem}_heatmap.ply"
        try:
            export_heatmap_ply(mesh, scores, ply_path)
            print(f"Heatmap PLY guardado: {ply_path}")
        except Exception as e:
            print(f"Erro ao guardar {ply_path}: {e}", file=sys.stderr)

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Ranking de importância de vértices (Quadric, Curvature, Normal Deviation)."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Ficheiro OBJ ou pasta com ficheiros OBJ",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Pasta de saída (default: ModelVerticeRanker1000/output)",
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=("json", "csv"),
        default="json",
        help="Formato de saída para ranking (quando input é ficheiro)",
    )
    parser.add_argument(
        "--ply",
        "-p",
        action="store_true",
        help="Guardar também heatmap PLY (cor por importância). Útil para ModelRetopology2000.",
    )
    args = parser.parse_args()

    inp = Path(args.input)
    if not inp.is_absolute():
        inp = inp.resolve()
    if not inp.exists():
        print(
            f"Input não encontrado: {inp}\n"
            f"  (Diretório atual: {Path.cwd()})",
            file=sys.stderr,
        )
        sys.exit(1)

    # Modo pasta: procurar recursivamente OBJs com "highpoly" no nome, gerar heatmap PLY na mesma pasta
    if inp.is_dir():
        highpoly_objs = [
            p for p in inp.rglob("*.obj")
            if "highpoly" in p.name.lower()
        ]
        if not highpoly_objs:
            print(
                f"Nenhum ficheiro .obj com 'highpoly' no nome encontrado em {inp}",
                file=sys.stderr,
            )
            sys.exit(1)
        ok = 0
        for obj_path in highpoly_objs:
            mesh = load_mesh(obj_path)
            if mesh is None:
                continue
            scores, _, _ = compute_importance_ranking(mesh)
            ply_path = obj_path.parent / f"{obj_path.stem}_heatmap.ply"
            try:
                export_heatmap_ply(mesh, scores, ply_path)
                print(f"Heatmap guardado: {ply_path}")
                ok += 1
            except Exception as e:
                print(f"Erro ao guardar {ply_path}: {e}", file=sys.stderr)
        print(f"Processados {ok}/{len(highpoly_objs)} ficheiros highpoly.")
        return

    # Modo ficheiro: comportamento original (ranking JSON/CSV para pasta de saída)
    out_dir = args.output or OUTPUT_DIR
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [inp]
    if inp.suffix.lower() not in (".obj", ".ply", ".stl"):
        print("Aviso: ficheiro pode não ser uma mesh suportada.", file=sys.stderr)
    ok = 0
    for p in paths:
        if run_on_mesh(p, out_dir, args.format, export_ply=args.ply):
            ok += 1
    print(f"Processados {ok}/{len(paths)} ficheiros.")


if __name__ == "__main__":
    main()
