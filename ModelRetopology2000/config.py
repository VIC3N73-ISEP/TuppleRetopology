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

# Modo avançado: mais pares para treino/validação
TRAIN_PAIRS = 4000
VAL_PAIRS = 400

# 10 features: 9 (pos, normal, curvatura, etc.) + 1 (importância do heatmap)
# Modo avançado: rede mais larga e profunda
MODEL_CONFIG = {
    "node_features": 10,
    "hidden_dim": 384,
    "num_layers": 10,
    "dropout": 0.25,
    "use_batch_norm": True,
}

# Modo avançado: mais épocas, weight decay, grad clip, early stopping
TRAINING_CONFIG = {
    "batch_size": 24,
    "learning_rate": 5e-5,
    "weight_decay": 1e-5,
    "num_epochs": 1000,
    "save_every": 5,
    "val_every": 2,
    "grad_clip_max_norm": 1.0,
    "early_stopping_patience": 80,  # 0 = desativado
    "device": "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
}

MESH_CONFIG = {
    "max_vertices": 100000,
    "normalize": True,
    "add_features": True,
}
