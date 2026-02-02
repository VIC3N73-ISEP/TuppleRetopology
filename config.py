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

# Configurações de treinamento
TRAIN_PAIRS = 9500  # X - número de pares para treino
VAL_PAIRS = 500     # Y - número de pares para validação/avaliação

# Configurações do modelo GNN (CONFIGURAÇÃO PESADA E COMPLEXA)
MODEL_CONFIG = {
    "node_features": 9,      # Features por vértice (posição, normal, etc.)
    "hidden_dim": 256,       # Dimensão oculta (aumentado para modelo mais complexo e poderoso)
    "num_layers": 8,         # Número de camadas GNN (mais profundidade = mais capacidade de aprendizado)
    "dropout": 0.2,          # Dropout rate (aumentado para evitar overfitting em modelo grande)
    "use_batch_norm": True,  # Usar batch normalization
}

# Configurações de treinamento (TREINAMENTO ROBUSTO E COMPLETO)
TRAINING_CONFIG = {
    "batch_size": 2,         # Tamanho do batch (reduzido para caber modelo grande na memória)
    "learning_rate": 0.0005, # Taxa de aprendizado (reduzida para treinamento mais estável e preciso)
    "num_epochs": 200,       # Número de épocas (mais épocas para convergência completa)
    "save_every": 10,        # Salvar checkpoint a cada N épocas
    "val_every": 5,          # Validar a cada N épocas
    "device": "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
}

# Configurações de processamento de mesh (MÁXIMA QUALIDADE)
MESH_CONFIG = {
    "max_vertices": 100000,  # Máximo de vértices (aumentado para processar meshes grandes e detalhados)
    "normalize": True,       # Normalizar meshes
    "add_features": True,    # Adicionar features extras (curvatura, etc.)
}

