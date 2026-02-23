"""
CLI para executar o ranker de importância de vértices em ficheiros OBJ.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import trimesh

from config import OUTPUT_DIR
from ranker import compute_importance_ranking


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
        help="Formato de saída: json ou csv",
    )
    args = parser.parse_args()

    out_dir = args.output or OUTPUT_DIR
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    inp = Path(args.input)
    if not inp.exists():
        print(f"Input não encontrado: {inp}", file=sys.stderr)
        sys.exit(1)

    paths = []
    if inp.is_file():
        if inp.suffix.lower() not in (".obj", ".ply", ".stl"):
            print("Aviso: ficheiro pode não ser uma mesh suportada.", file=sys.stderr)
        paths.append(inp)
    else:
        for ext in ("*.obj", "*.OBJ", "*.ply", "*.stl"):
            paths.extend(inp.glob(ext))

    if not paths:
        print("Nenhum ficheiro de mesh encontrado.", file=sys.stderr)
        sys.exit(1)

    ok = 0
    for p in paths:
        if run_on_mesh(p, out_dir, args.format):
            ok += 1
    print(f"Processados {ok}/{len(paths)} ficheiros.")


if __name__ == "__main__":
    main()
