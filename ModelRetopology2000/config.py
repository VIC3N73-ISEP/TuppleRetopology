"""
Configurações do ModelRetopology2000.
Modelo que usa heatmap (vértices importantes), highpoly e lowpoly para aprender a transformar highpoly em lowpoly.
"""

import os
from pathlib import Path

# Diretórios
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "Expai-Retopo"
ORGANIZED_PAIRS_DIR = PROJECT_ROOT.parent / "trains_single_pair"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
OUTPUT_DIR = PROJECT_ROOT / "output"

CHECKPOINTS_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

TRAIN_PAIRS = 3800
VAL_PAIRS = 200

# 10 features: 9 (pos, normal, curvatura, etc.) + 1 (importância do heatmap)
MODEL_CONFIG = {
    "node_features": 10,
    "hidden_dim": 256,
    "num_layers": 8,
    "dropout": 0.2,
    "use_batch_norm": True,
}

TRAINING_CONFIG = {
    "batch_size": 30,
    "learning_rate": 0.0001,
    "num_epochs": 500,
    "save_every": 10,
    "val_every": 5,
    "device": "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
}

MESH_CONFIG = {
    "max_vertices": 100000,
    "normalize": True,
    "add_features": True,
}
