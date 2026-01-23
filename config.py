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
    "hidden_dim": 64,        # Dimensão oculta (reduzido de 128 para acelerar)
    "num_layers": 4,         # Número de camadas GNN (reduzido de 6 para acelerar)
    "dropout": 0.1,          # Dropout rate
    "use_batch_norm": True,  # Usar batch normalization
}

# Configurações de treinamento
TRAINING_CONFIG = {
    "batch_size": 8,         # Tamanho do batch (aumentado de 4 para acelerar)
    "learning_rate": 0.002,  # Taxa de aprendizado (aumentada para convergir mais rápido)
    "num_epochs": 50,        # Número de épocas (reduzido de 100 para testes rápidos)
    "save_every": 10,        # Salvar checkpoint a cada N épocas
    "val_every": 5,          # Validar a cada N épocas
    "device": "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
}

# Configurações de processamento de mesh
MESH_CONFIG = {
    "max_vertices": 20000,   # Máximo de vértices (reduzido de 50000 para acelerar processamento)
    "normalize": True,       # Normalizar meshes
    "add_features": True,    # Adicionar features extras (curvatura, etc.)
}

