"""
Inferência ModelRetopology3000: prevê ratio (opcional), aplica decimate e exporta OBJ.
"""

import torch
import trimesh
from pathlib import Path
import argparse
import subprocess
import tempfile
import textwrap

from torch_geometric.data import Batch

from config import MODEL_CONFIG, CHECKPOINTS_DIR, OUTPUT_DIR, RATIO_APPLY_MODE
from dataset import RetopologyDataset3000
from model import RetopologyModel3000


def _resolve_device(requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    if not torch.cuda.is_available():
        return "cpu"
    try:
        torch.zeros(1, device="cuda")
        return "cuda"
    except RuntimeError as e:
        if "no kernel image" in str(e) or "not compatible" in str(e).lower():
            print("⚠️  GPU não suportada por esta build do PyTorch. A usar CPU.")
            return "cpu"
        raise


def load_model(checkpoint_path, device="cpu"):
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt.get("model_config", MODEL_CONFIG)
    model = RetopologyModel3000(cfg)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.to(device)
    model.eval()
    return model, ckpt


def decimate_with_blender(input_obj: Path, output_obj: Path, ratio: float, blender_exec: str = "blender"):
    """
    Aplica Blender Decimate Modifier exatamente via bpy:
    - mode: COLLAPSE
    - use_collapse_triangulate: False
    - use_symmetry: False
    """
    script = textwrap.dedent(
        f"""
        import bpy
        import sys

        input_obj = r\"{str(input_obj)}\"
        output_obj = r\"{str(output_obj)}\"
        ratio = float({ratio})

        bpy.ops.wm.read_factory_settings(use_empty=True)

        bpy.ops.wm.obj_import(filepath=input_obj)
        objs = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
        if not objs:
            raise RuntimeError("Nenhum objeto MESH importado do OBJ.")

        # Processar todos os meshes importados
        for obj in objs:
            bpy.context.view_layer.objects.active = obj
            obj.select_set(True)

            mod = obj.modifiers.new(name="Decimate", type='DECIMATE')
            mod.decimate_type = 'COLLAPSE'
            mod.ratio = ratio
            mod.use_collapse_triangulate = False
            mod.use_symmetry = False

            bpy.ops.object.modifier_apply(modifier=mod.name)
            obj.select_set(False)

        bpy.ops.object.select_all(action='DESELECT')
        for obj in objs:
            obj.select_set(True)
        bpy.context.view_layer.objects.active = objs[0]
        bpy.ops.wm.obj_export(filepath=output_obj, export_selected_objects=True)
        """
    )

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as tmp:
        tmp.write(script)
        script_path = Path(tmp.name)

    try:
        cmd = [blender_exec, "--background", "--python", str(script_path)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                "Falha ao executar Blender Decimate.\n"
                f"Comando: {' '.join(cmd)}\n"
                f"stdout:\n{proc.stdout}\n"
                f"stderr:\n{proc.stderr}"
            )
    finally:
        try:
            script_path.unlink(missing_ok=True)
        except Exception:
            pass


def predict_retopology(
    obj_path,
    checkpoint_path=None,
    ratio=None,
    output_path=None,
    device="cpu",
    model=None,
    blender_exec="blender",
):
    """
    ratio: None = prever automaticamente; float [0,1] = forçar ratio final previsto.
    O valor aplicado ao decimate é (1 - ratio_final).
    """
    obj_path = Path(obj_path)
    if not obj_path.exists():
        raise FileNotFoundError(f"OBJ não encontrado: {obj_path}")

    if model is None:
        if checkpoint_path is None:
            checkpoint_path = CHECKPOINTS_DIR / "best_model.pt"
        if not Path(checkpoint_path).exists():
            checkpoint_path = CHECKPOINTS_DIR / "last_checkpoint.pt"
        model, _ = load_model(checkpoint_path, device)

    temp_ds = RetopologyDataset3000(Path("."), [], config=MODEL_CONFIG)
    mesh = trimesh.load(str(obj_path))
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.geometry[list(mesh.geometry.keys())[0]]
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("Ficheiro não é uma malha Trimesh válida.")
    high_graph = temp_ds.mesh_to_graph_high(mesh)
    high_batch = Batch.from_data_list([high_graph]).to(device)

    with torch.no_grad():
        if ratio is not None:
            r = float(max(0.0, min(1.0, ratio)))
        else:
            r = model.predict_ratio(high_batch).item()

    if RATIO_APPLY_MODE == "one_minus":
        decimate_ratio = 1.0 - r
    else:
        decimate_ratio = r

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        decimate_with_blender(obj_path, output_path, decimate_ratio, blender_exec=blender_exec)
        print(
            f"Guardado (decimate-only, Blender): {output_path} "
            f"(ratio final: {r:.4f}, ratio aplicado no decimate: {decimate_ratio:.4f}, "
            f"modo: {RATIO_APPLY_MODE})"
        )
    return output_path, r


def main():
    parser = argparse.ArgumentParser(
        description="ModelRetopology3000: decimate-only (ratio omitido = automático)"
    )
    parser.add_argument("--input", "-i", type=Path, required=True, help="OBJ de entrada (high poly)")
    parser.add_argument(
        "--ratio",
        "-r",
        type=float,
        default=None,
        help="Ratio Decimate [0,1] (0=pouca redução, 1=máxima). Omitir = prever automaticamente.",
    )
    parser.add_argument("--checkpoint", "-c", type=Path, default=None, help="Checkpoint .pt")
    parser.add_argument("--output", "-o", type=Path, default=None, help="OBJ de saída")
    parser.add_argument("--device", default="cpu", help="cpu ou cuda")
    parser.add_argument(
        "--blender-exec",
        type=str,
        default="blender",
        help="Executável do Blender (ex.: blender ou caminho completo).",
    )
    args = parser.parse_args()

    device = _resolve_device(args.device)
    print(f"Dispositivo: {device}")

    out = args.output
    if out is None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out = OUTPUT_DIR / (args.input.stem + "_retopology3000.obj")

    _, r = predict_retopology(
        args.input,
        checkpoint_path=args.checkpoint,
        ratio=args.ratio,
        output_path=out,
        device=device,
        blender_exec=args.blender_exec,
    )
    print(f"Ratio final: {r:.6f}")


if __name__ == "__main__":
    main()
