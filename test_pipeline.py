"""
Script de teste rápido para validar a pipeline.
Verifica se modelo e loss estão funcionando corretamente.
"""

import torch
from torch_geometric.data import Data
from config import MODEL_CONFIG
from model import RetopologyGNN as Model
from train import RetopologyLoss as Loss

print("[OK] Modelo e loss carregados")

# Criar dados de teste
print("\nCriando dados de teste...")
num_nodes = 100
num_features = MODEL_CONFIG["node_features"]

# Criar grafo de teste
x = torch.randn(num_nodes, num_features)
edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))  # Grafo aleatório
edge_index = edge_index[:, edge_index[0] != edge_index[1]]  # Remover self-loops

test_data = Data(x=x, edge_index=edge_index, pos=x[:, :3])

# Criar modelo
print("Criando modelo...")
model = Model(MODEL_CONFIG)
print(f"[OK] Modelo criado com {sum(p.numel() for p in model.parameters())} parâmetros")

# Forward pass
print("\nTestando forward pass...")
model.eval()
with torch.no_grad():
    pred_positions, pred_features = model(test_data)
    print(f"[OK] Forward pass OK")
    print(f"  - Pred positions shape: {pred_positions.shape}")
    print(f"  - Pred features shape: {pred_features.shape}")
    print(f"  - Pred positions range: [{pred_positions.min().item():.3f}, {pred_positions.max().item():.3f}]")
    print(f"  - Pred features range: [{pred_features.min().item():.3f}, {pred_features.max().item():.3f}]")

# Criar target
target_positions = torch.randn(50, 3)  # Diferente número de vértices (teste Chamfer)
target_features = torch.randn(50, 6)

# Criar loss
print("\nTestando loss...")
criterion = Loss()
model.train()

# Forward pass com gradiente
pred_positions, pred_features = model(test_data)
print(f"  - Pred positions shape: {pred_positions.shape}")
print(f"  - Target positions shape: {target_positions.shape}")

# Calcular loss
loss, loss_dict = criterion(
    pred_positions=pred_positions,
    pred_features=pred_features,
    target_positions=target_positions,
    target_features=target_features,
    pred_edge_index=test_data.edge_index,
    target_edge_index=test_data.edge_index
)

print(f"[OK] Loss calculada")
print(f"  - Total loss: {loss.item():.6f}")
print(f"  - Loss tem gradiente: {loss.requires_grad}")
print(f"  - Componentes da loss:")
for key, value in loss_dict.items():
    print(f"    - {key}: {value:.6f}")

# Verificar se loss não é zero
if loss.item() == 0.0:
    print("\n[AVISO] Loss total é zero! Isso pode indicar um problema.")
else:
    print("\n[OK] Loss não é zero - pipeline funcionando corretamente!")

# Testar backward
print("\nTestando backward pass...")
loss.backward()
print("[OK] Backward pass OK")

# Verificar gradientes
has_grad = False
for name, param in model.named_parameters():
    if param.grad is not None:
        has_grad = True
        break

if has_grad:
    print("[OK] Gradientes calculados corretamente")
else:
    print("[AVISO] Nenhum gradiente calculado!")

print("\n" + "="*70)
print("Teste concluído!")
print("="*70)
