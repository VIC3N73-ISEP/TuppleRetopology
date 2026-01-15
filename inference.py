"""
Script de inferência: gera low poly mesh a partir de high poly mesh usando o modelo treinado.
"""

import torch
import numpy as np
import trimesh
from pathlib import Path
import argparse

from config import MODEL_CONFIG, CHECKPOINTS_DIR, OUTPUT_DIR
from model import RetopologyGNN, ImprovedRetopologyGNN
from dataset import RetopologyDataset


def load_model(checkpoint_path, device="cpu"):
    """
    Carrega o modelo a partir de um checkpoint.
    
    Args:
        checkpoint_path: Caminho para o checkpoint
        device: Dispositivo (cpu/cuda)
        
    Returns:
        Modelo carregado
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Obter configuração do checkpoint ou usar padrão
    model_config = checkpoint.get("model_config", MODEL_CONFIG)
    
    # Criar modelo
    model = RetopologyGNN(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    return model, checkpoint


def mesh_to_graph(mesh, config=None):
    """
    Converte um mesh para grafo (similar ao dataset).
    """
    from dataset import RetopologyDataset
    
    if config is None:
        config = MODEL_CONFIG
    
    # Criar dataset temporário para usar o método mesh_to_graph
    temp_dataset = RetopologyDataset(Path("."), [], config)
    return temp_dataset.mesh_to_graph(mesh)


def graph_to_mesh(graph_data, output_path):
    """
    Converte um grafo de volta para mesh OBJ.
    
    Args:
        graph_data: Data object com posições e features
        output_path: Caminho para salvar o OBJ
    """
    positions = graph_data.pos.cpu().numpy()
    
    # Criar mesh básico a partir das posições
    # Usar Delaunay triangulation ou convex hull como fallback
    try:
        # Tentar criar mesh usando convex hull
        mesh = trimesh.Trimesh(vertices=positions)
        
        # Se não houver faces, criar usando convex hull
        if len(mesh.faces) == 0:
            mesh = trimesh.convex.convex_hull(positions)
        
        # Salvar mesh
        mesh.export(str(output_path))
        print(f"✓ Mesh salvo em {output_path}")
        print(f"  - Vértices: {len(mesh.vertices)}")
        print(f"  - Faces: {len(mesh.faces)}")
        
        return mesh
        
    except Exception as e:
        print(f"Erro ao criar mesh: {e}")
        # Salvar apenas vértices como ponto cloud
        point_cloud = trimesh.PointCloud(vertices=positions)
        point_cloud.export(str(output_path))
        print(f"✓ Ponto cloud salvo em {output_path}")
        return point_cloud


def predict_lowpoly(highpoly_path, checkpoint_path, output_path=None, target_vertices=None, device="cpu"):
    """
    Gera um low poly mesh a partir de um high poly mesh.
    
    Args:
        highpoly_path: Caminho para o arquivo OBJ high poly
        checkpoint_path: Caminho para o checkpoint do modelo
        output_path: Caminho para salvar o resultado (opcional)
        target_vertices: Número desejado de vértices (opcional)
        device: Dispositivo (cpu/cuda)
        
    Returns:
        Mesh do low poly gerado
    """
    print("=" * 70)
    print("Inferência: Gerando Low Poly a partir de High Poly")
    print("=" * 70)
    print()
    
    # Carregar modelo
    print(f"Carregando modelo de {checkpoint_path}...")
    model, checkpoint = load_model(checkpoint_path, device)
    print(f"✓ Modelo carregado (época {checkpoint.get('epoch', '?')})")
    print()
    
    # Carregar high poly mesh
    print(f"Carregando high poly mesh de {highpoly_path}...")
    highpoly_mesh = trimesh.load(str(highpoly_path))
    
    # Se for Scene, pegar primeiro mesh
    if isinstance(highpoly_mesh, trimesh.Scene):
        highpoly_mesh = highpoly_mesh.geometry[list(highpoly_mesh.geometry.keys())[0]]
    
    print(f"✓ High poly carregado")
    print(f"  - Vértices: {len(highpoly_mesh.vertices)}")
    print(f"  - Faces: {len(highpoly_mesh.faces)}")
    print()
    
    # Converter para grafo
    print("Convertendo mesh para grafo...")
    highpoly_graph = mesh_to_graph(highpoly_mesh)
    print(f"✓ Grafo criado")
    print(f"  - Nós: {highpoly_graph.x.shape[0]}")
    print(f"  - Arestas: {highpoly_graph.edge_index.shape[1]}")
    print()
    
    # Fazer predição
    print("Gerando low poly...")
    highpoly_batch = highpoly_graph.to(device)
    
    with torch.no_grad():
        pred_positions, pred_features = model(highpoly_batch)
    
    # Criar grafo do low poly
    lowpoly_graph = highpoly_graph.clone()
    lowpoly_graph.pos = pred_positions.cpu()
    lowpoly_graph.x = torch.cat([pred_positions.cpu(), pred_features.cpu()], dim=1)
    
    # Reduzir número de vértices se necessário
    if target_vertices and len(lowpoly_graph.pos) > target_vertices:
        print(f"Reduzindo para {target_vertices} vértices...")
        # Selecionar vértices mais importantes
        importance = torch.norm(lowpoly_graph.pos, dim=1)
        _, indices = torch.topk(importance, target_vertices)
        lowpoly_graph.pos = lowpoly_graph.pos[indices]
        lowpoly_graph.x = lowpoly_graph.x[indices]
        print(f"✓ Reduzido para {len(lowpoly_graph.pos)} vértices")
    
    print(f"✓ Low poly gerado")
    print(f"  - Vértices: {len(lowpoly_graph.pos)}")
    print()
    
    # Salvar resultado
    if output_path is None:
        output_path = OUTPUT_DIR / f"lowpoly_{Path(highpoly_path).stem}.obj"
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Salvando resultado em {output_path}...")
    lowpoly_mesh = graph_to_mesh(lowpoly_graph, output_path)
    
    print()
    print("=" * 70)
    print("Inferência concluída!")
    print("=" * 70)
    
    return lowpoly_mesh


def main():
    """
    Função principal de inferência.
    """
    parser = argparse.ArgumentParser(description="Gerar low poly mesh a partir de high poly")
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Caminho para o arquivo OBJ high poly"
    )
    parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        default=None,
        help="Caminho para o checkpoint (padrão: melhor modelo)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Caminho para salvar o resultado (padrão: output/lowpoly_<nome>.obj)"
    )
    parser.add_argument(
        "--target-vertices", "-v",
        type=int,
        default=None,
        help="Número desejado de vértices no low poly (opcional)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Dispositivo a usar (cpu/cuda)"
    )
    
    args = parser.parse_args()
    
    # Determinar checkpoint
    if args.checkpoint is None:
        checkpoint_path = CHECKPOINTS_DIR / "best_model.pt"
        if not checkpoint_path.exists():
            checkpoint_path = CHECKPOINTS_DIR / "last_checkpoint.pt"
    else:
        checkpoint_path = Path(args.checkpoint)
    
    if not checkpoint_path.exists():
        print(f"❌ Erro: Checkpoint não encontrado em {checkpoint_path}")
        return
    
    # Verificar dispositivo
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA não disponível, usando CPU")
        device = "cpu"
    
    # Executar inferência
    predict_lowpoly(
        args.input,
        checkpoint_path,
        args.output,
        args.target_vertices,
        device
    )


if __name__ == "__main__":
    main()

