"""
Configurações do projeto para treinamento de IA de retopologia.
"""

import os
from pathlib import Path

# Diretórios
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "Expai-Retopo"
ORGANIZED_PAIRS_DIR = PROJECT_ROOT / "../trains_single_pair"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Criar diretórios se não existirem
CHECKPOINTS_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Configurações de treinamento (modo avançado: mais dados, mais épocas)
TRAIN_PAIRS = 4000   # usar mais pares para treino
VAL_PAIRS = 400      # mais validação para estimativa estável

# Configurações do modelo GNN (modo avançado: rede mais profunda e larga)
MODEL_CONFIG = {
    "node_features": 9,       # Features por vértice (posição, normal, etc.)
    "hidden_dim": 384,        # Dimensão oculta maior (mais capacidade)
    "num_layers": 10,         # Mais camadas GNN (mais profundidade)
    "dropout": 0.25,         # Dropout um pouco maior para regularização
    "use_batch_norm": True,  # Batch normalization
}

# Configurações de treinamento (modo avançado: mais épocas, weight decay, grad clip)
TRAINING_CONFIG = {
    "batch_size": 24,              # batch menor se GPU tiver pouca memória com modelo maior
    "learning_rate": 5e-5,         # LR mais baixo para treino longo estável
    "weight_decay": 1e-5,          # L2 regularization
    "num_epochs": 1000,             # treino longo
    "save_every": 5,                # checkpoints mais frequentes
    "val_every": 2,                 # validação frequente
    "grad_clip_max_norm": 1.0,      # clip de gradientes (estabilidade)
    "early_stopping_patience": 80,  # parar se val_loss não melhorar em N épocas (0 = desativado)
    "device": "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
}

# Configurações de processamento de mesh (MÁXIMA QUALIDADE)
MESH_CONFIG = {
    "max_vertices": 100000,  # Máximo de vértices (aumentado para processar meshes grandes e detalhados)
    "normalize": True,       # Normalizar meshes
    "add_features": True,   # Adicionar features extras (curvatura, etc.)
}
