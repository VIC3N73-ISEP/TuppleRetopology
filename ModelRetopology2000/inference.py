"""
Inferência ModelRetopology2000: recebe heatmap PLY + highpoly, gera lowpoly.
"""

import torch
import trimesh
from pathlib import Path
import argparse

from config import MODEL_CONFIG, CHECKPOINTS_DIR, OUTPUT_DIR


def _resolve_device(requested: str) -> str:
    """Usa CUDA só se estiver disponível e esta build do PyTorch suportar a GPU (ex.: RTX 50xx = sm_120 ainda não suportado)."""
    if requested == "cpu":
        return "cpu"
    if not torch.cuda.is_available():
        return "cpu"
    try:
        torch.zeros(1, device="cuda")
        return "cuda"
    except RuntimeError as e:
        if "no kernel image" in str(e) or "not compatible" in str(e).lower():
            print("⚠️  GPU não suportada por esta build do PyTorch (ex.: RTX 50xx). A usar CPU.")
            return "cpu"
        raise
from model import RetopologyGNN2000
from dataset import RetopologyDataset2000, _importance_from_heatmap_ply


def load_model(checkpoint_path, device="cpu"):
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt.get("model_config", MODEL_CONFIG)
    model = RetopologyGNN2000(cfg)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.to(device)
    model.eval()
    return model, ckpt


def predict_lowpoly(highpoly_path, heatmap_path, checkpoint_path=None, output_path=None, device="cpu", model=None):
    """
    Gera lowpoly a partir de highpoly + heatmap PLY.
    highpoly_path: OBJ highpoly
    heatmap_path: PLY com vertex colors = importância (vermelho = importante)
    """
    if model is None:
        if checkpoint_path is None:
            checkpoint_path = CHECKPOINTS_DIR / "best_model.pt"
        if not Path(checkpoint_path).exists():
            checkpoint_path = CHECKPOINTS_DIR / "last_checkpoint.pt"
        model, _ = load_model(checkpoint_path, device)

    highpoly_path = Path(highpoly_path)
    heatmap_path = Path(heatmap_path)
    if not highpoly_path.exists():
        raise FileNotFoundError(f"Highpoly não encontrado: {highpoly_path}")
    if not heatmap_path.exists():
        raise FileNotFoundError(f"Heatmap PLY não encontrado: {heatmap_path}")

    highpoly_mesh = trimesh.load(str(highpoly_path))
    if isinstance(highpoly_mesh, trimesh.Scene):
        highpoly_mesh = highpoly_mesh.geometry[list(highpoly_mesh.geometry.keys())[0]]

    importance = _importance_from_heatmap_ply(heatmap_path, len(highpoly_mesh.vertices))
    temp_ds = RetopologyDataset2000(Path("."), [], config=MODEL_CONFIG)
    highpoly_graph = temp_ds.mesh_to_graph(highpoly_mesh, importance=importance)
    from torch_geometric.data import Batch
    highpoly_batch = Batch.from_data_list([highpoly_graph]).to(device)

    with torch.no_grad():
        pred_positions, pred_features = model(highpoly_batch)

    positions = pred_positions.cpu().numpy()
    mesh_out = trimesh.Trimesh(vertices=positions)
    if len(mesh_out.faces) == 0 and len(positions) >= 3:
        try:
            mesh_out = trimesh.convex.convex_hull(positions)
        except Exception:
            mesh_out = trimesh.PointCloud(vertices=positions)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mesh_out.export(str(output_path))
        print(f"Lowpoly guardado: {output_path}")
    return mesh_out


def main():
    parser = argparse.ArgumentParser(description="Inferência ModelRetopology2000: heatmap + highpoly -> lowpoly")
    parser.add_argument("--highpoly", "-i", type=Path, required=True, help="Ficheiro OBJ highpoly")
    parser.add_argument("--heatmap", "-m", type=Path, required=True, dest="heatmap", help="Ficheiro PLY heatmap (vértices importantes)")
    parser.add_argument("--output", "-o", type=Path, default=None, help="Ficheiro de saída (OBJ/PLY)")
    parser.add_argument("--checkpoint", "-c", type=Path, default=None, help="Checkpoint (default: best_model.pt)")
    parser.add_argument("--device", default="cpu", help="cpu ou cuda")
    args = parser.parse_args()

    device = _resolve_device(args.device)
    print(f"Dispositivo: {device}")

    out = args.output or OUTPUT_DIR / (args.highpoly.stem + "_lowpoly.obj")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    predict_lowpoly(
        args.highpoly,
        args.heatmap,
        checkpoint_path=args.checkpoint,
        output_path=out,
        device=device,
    )


if __name__ == "__main__":
    main()
