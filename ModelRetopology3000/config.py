"""
Configurações do ModelRetopology3000.
Pipeline decimate-only: o modelo aprende a prever o ratio.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "Expai-Retopo"
ORGANIZED_PAIRS_DIR = PROJECT_ROOT.parent / "train_mass_pair"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
OUTPUT_DIR = PROJECT_ROOT / "output"

CHECKPOINTS_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Semântica global do ratio:
# - modelo prevê ratio_final
# - decimate aplicado usa ratio_applied
# - com modo "one_minus": ratio_applied = 1 - ratio_final
RATIO_APPLY_MODE = "one_minus"

TRAIN_PAIRS = 4000
VAL_PAIRS = 400

# 9 = pos, normal, curvatura, dist_centro, placeholder (igual ao 2000 sem heatmap)
BASE_NODE_FEATURES = 9
MODEL_CONFIG = {
    "base_node_features": BASE_NODE_FEATURES,
    "node_features": BASE_NODE_FEATURES + 1,  # mantido para compatibilidade de checkpoints antigos
    "hidden_dim": 384,
    "num_layers": 10,
    "dropout": 0.25,
    "use_batch_norm": True,
    # Cabeça de ratio: encoder mais curto
    "ratio_gcn_layers": 3,
}

TRAINING_CONFIG = {
    "batch_size": 24,
    "learning_rate": 5e-5,
    "weight_decay": 1e-5,
    "num_epochs": 1000,
    "save_every": 5,
    "val_every": 2,
    "grad_clip_max_norm": 1.0,
    "early_stopping_patience": 80,
    "device": "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
}

LOSS_CONFIG = {
    "w_ratio": 1.0,
}

MESH_CONFIG = {
    "max_vertices": 100000,
    "normalize": True,
    "add_features": True,
}

# Evita divisão por zero ao calcular r_gt a partir de F_high / F_low
EPS_FACE = 1e-6


def ratio_to_target_face_count(n_faces: int, r: float) -> int:
    """
    r in [0,1]: 0 ~ manter faces, 1 ~ redução máxima.
    target_faces = max(4, round(n_faces * (1 - r))).
    """
    r = float(max(0.0, min(1.0, r)))
    n = int(n_faces)
    if n <= 0:
        return 4
    target = max(4, int(round(n * (1.0 - r))))
    return min(target, n)


def r_gt_from_face_counts(n_faces_high: int, n_faces_low: int) -> float:
    """
    Rótulo de ratio_final a partir do par ground truth: 1 - F_low/F_high (clip [0,1]).
    Para inferência com RATIO_APPLY_MODE='one_minus', o valor aplicado é:
    ratio_applied = 1 - ratio_final.
    """
    nh = max(int(n_faces_high), 1)
    nl = int(n_faces_low)
    ratio = 1.0 - (nl + EPS_FACE) / (nh + EPS_FACE)
    return float(max(0.0, min(1.0, ratio)))
