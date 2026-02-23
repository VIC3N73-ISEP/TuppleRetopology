"""
Abre um ficheiro PLY (ou OBJ) com cores por vértice como point cloud.
Usa trimesh + matplotlib (sem precisar de Open3D).
Uso: python VisualizationTools/view_colored_ply.py ficheiro_colored.ply
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import trimesh


def main():
    parser = argparse.ArgumentParser(
        description="Ver point cloud colorida (heatmap) com trimesh + matplotlib."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Ficheiro PLY ou OBJ com vertex colors (ex.: highpoly-0013_colored.ply)",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=1.0,
        help="Tamanho dos pontos (default: 1.0)",
    )
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        print(f"Ficheiro não encontrado: {path}", file=sys.stderr)
        sys.exit(1)

    mesh = trimesh.load(str(path))
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.geometry[list(mesh.geometry.keys())[0]]

    vertices = np.asarray(mesh.vertices)
    if hasattr(mesh.visual, "vertex_colors") and mesh.visual.vertex_colors is not None:
        colors = np.asarray(mesh.visual.vertex_colors[:, :3], dtype=np.float64) / 255.0
    else:
        colors = np.ones((len(vertices), 3)) * 0.7

    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("matplotlib não está instalado. Instale com:  pip install matplotlib", file=sys.stderr)
        sys.exit(1)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        vertices[:, 0],
        vertices[:, 1],
        vertices[:, 2],
        c=colors,
        s=args.point_size,
        marker=".",
        edgecolors="none",
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Heatmap (point cloud)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
