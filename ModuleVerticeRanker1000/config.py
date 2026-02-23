"""
Configurações do ModuleVerticeRanker1000.
O módulo não usa número de vértices inicial/final: conta os vértices da malha
introduzida e faz o ranking para todos (do mais ao menos importante).
"""

from pathlib import Path

# Diretórios
PROJECT_ROOT = Path(__file__).parent
OUTPUT_DIR = PROJECT_ROOT / "output"

OUTPUT_DIR.mkdir(exist_ok=True)

# Pesos das métricas para o score final (quadric, curvature, normal_deviation)
VERTICE_RANKER_WEIGHTS = {
    "quadric": 0.4,
    "curvature": 0.3,
    "normal_deviation": 0.3,
}

# Opções de pré-processamento da mesh (apenas normalizar escala/centro; sem limite de vértices)
MESH_CONFIG = {
    "normalize": False,
}
