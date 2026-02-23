"""
Configurações do projeto para treinamento de IA de retopologia.
"""

import os
from pathlib import Path

# Diretórios
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "Expai-Retopo"
ORGANIZED_PAIRS_DIR = PROJECT_ROOT / "trains_single_pair"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Criar diretórios se não existirem
CHECKPOINTS_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Configurações de treinamento (modo rápido para modelo normal)
TRAIN_PAIRS = 1900  # treino rápido
VAL_PAIRS = 100     # validação rápida

# Configurações do modelo GNN (CONFIGURAÇÃO PESADA E COMPLEXA)
MODEL_CONFIG = {
    "node_features": 9,      # Features por vértice (posição, normal, etc.)
    "hidden_dim": 256,       # Dimensão oculta (aumentado para modelo mais complexo e poderoso)
    "num_layers": 8,         # Número de camadas GNN (mais profundidade = mais capacidade de aprendizado)
    "dropout": 0.2,          # Dropout rate (aumentado para evitar overfitting em modelo grande)
    "use_batch_norm": True,  # Usar batch normalization
}

# Configurações de treinamento (modo rápido para train.py)
TRAINING_CONFIG = {
    "batch_size": 30,         # maior batch para acelerar iterações
    "learning_rate": 0.0001,   # valor estável para treino curto
    "num_epochs": 200,        # treino rápido
    "save_every": 10,         # salvar mais frequente no treino curto
    "val_every": 5,          # validação frequente para acompanhar convergência
    "device": "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
}

# Configurações de processamento de mesh (MÁXIMA QUALIDADE)
MESH_CONFIG = {
    "max_vertices": 100000,  # Máximo de vértices (aumentado para processar meshes grandes e detalhados)
    "normalize": True,       # Normalizar meshes
    "add_features": True,   # Adicionar features extras (curvatura, etc.)
}
