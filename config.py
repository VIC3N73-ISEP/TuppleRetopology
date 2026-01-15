"""
Configurações do projeto para treinamento de IA de retopologia.
"""

import os
from pathlib import Path

# Diretórios
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "Expai-Retopo"
ORGANIZED_PAIRS_DIR = PROJECT_ROOT / "training_data"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Criar diretórios se não existirem
CHECKPOINTS_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Configurações de treinamento
TRAIN_PAIRS = 20  # X - número de pares para treino
VAL_PAIRS = 5     # Y - número de pares para validação/avaliação

# Configurações do modelo GNN
MODEL_CONFIG = {
    "node_features": 9,      # Features por vértice (posição, normal, etc.)
    "hidden_dim": 128,       # Dimensão oculta
    "num_layers": 6,         # Número de camadas GNN
    "dropout": 0.1,          # Dropout rate
    "use_batch_norm": True,  # Usar batch normalization
}

# Configurações de treinamento
TRAINING_CONFIG = {
    "batch_size": 4,         # Tamanho do batch
    "learning_rate": 0.001,  # Taxa de aprendizado
    "num_epochs": 100,       # Número de épocas
    "save_every": 10,        # Salvar checkpoint a cada N épocas
    "val_every": 5,          # Validar a cada N épocas
    "device": "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
}

# Configurações de processamento de mesh
MESH_CONFIG = {
    "max_vertices": 50000,   # Máximo de vértices (para limitar memória)
    "normalize": True,       # Normalizar meshes
    "add_features": True,    # Adicionar features extras (curvatura, etc.)
}

