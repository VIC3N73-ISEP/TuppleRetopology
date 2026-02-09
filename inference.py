"""
Script de inferência: gera low poly mesh a partir de high poly mesh usando o modelo treinado.
"""

import torch
import numpy as np
import trimesh
from pathlib import Path
import argparse

from config import MODEL_CONFIG, CHECKPOINTS_DIR, OUTPUT_DIR, ORGANIZED_PAIRS_DIR

# Importar modelo: tentar novo, fallback para antigo
try:
    from model_v2 import StableRetopologyGNN as RetopologyGNN
    USE_NEW_MODEL = True
except ImportError:
    from model import RetopologyGNN
    USE_NEW_MODEL = False

from dataset import RetopologyDataset


def load_model(checkpoint_path, device="cpu"):
    """
    Carrega o modelo a partir de um checkpoint.
    Compatível com checkpoints antigos e novos.
    
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
    model = RetopologyGNN(model_config, target_vertex_count=None) if USE_NEW_MODEL else RetopologyGNN(model_config)
    
    # Carregar state_dict com tratamento de chaves faltantes (compatibilidade com checkpoints antigos)
    state_dict = checkpoint["model_state_dict"]
    model_state_dict = model.state_dict()
    
    # Filtrar chaves que existem no modelo atual
    filtered_state_dict = {}
    missing_keys = []
    unexpected_keys = []
    
    for key in state_dict.keys():
        if key in model_state_dict:
            # Verificar se as dimensões são compatíveis
            if state_dict[key].shape == model_state_dict[key].shape:
                filtered_state_dict[key] = state_dict[key]
            else:
                print(f"⚠️  Aviso: Dimensões incompatíveis para {key}: checkpoint {state_dict[key].shape} vs modelo {model_state_dict[key].shape}")
                missing_keys.append(key)
        else:
            unexpected_keys.append(key)
    
    # Adicionar chaves que faltam no checkpoint (serão inicializadas com valores padrão)
    for key in model_state_dict.keys():
        if key not in filtered_state_dict:
            missing_keys.append(key)
    
    # Carregar o state_dict filtrado
    model_state_dict.update(filtered_state_dict)
    model.load_state_dict(model_state_dict, strict=False)
    
    if missing_keys:
        print(f"⚠️  Chaves faltantes no checkpoint (serão inicializadas): {len(missing_keys)}")
        if len(missing_keys) <= 10:  # Mostrar apenas se não for muitas
            for key in missing_keys[:5]:
                print(f"    - {key}")
            if len(missing_keys) > 5:
                print(f"    ... e mais {len(missing_keys) - 5}")
    
    if unexpected_keys:
        print(f"⚠️  Chaves inesperadas no checkpoint (ignoradas): {len(unexpected_keys)}")
        if len(unexpected_keys) <= 10:
            for key in unexpected_keys[:5]:
                print(f"    - {key}")
            if len(unexpected_keys) > 5:
                print(f"    ... e mais {len(unexpected_keys) - 5}")
    
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


def predict_lowpoly(highpoly_path, checkpoint_path=None, output_path=None, target_vertices=None, device="cpu", model=None):
    """
    Gera um low poly mesh a partir de um high poly mesh.
    
    Args:
        highpoly_path: Caminho para o arquivo OBJ high poly
        checkpoint_path: Caminho para o checkpoint do modelo (obrigatório se model=None)
        output_path: Caminho para salvar o resultado (opcional)
        target_vertices: Número desejado de vértices (opcional)
        device: Dispositivo (cpu/cuda)
        model: Modelo pré-carregado (opcional, para reutilização)
        
    Returns:
        Mesh do low poly gerado
    """
    print("=" * 70)
    print("Inferência: Gerando Low Poly a partir de High Poly")
    print("=" * 70)
    print()
    
    # Carregar modelo (se não foi fornecido)
    if model is None:
        if checkpoint_path is None:
            raise ValueError("checkpoint_path é obrigatório se model não for fornecido")
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
        if USE_NEW_MODEL:
            # Novo modelo retorna 4 valores: positions, features, importance_scores, selected_indices
            # Se target_vertices for especificado, usar no forward
            if target_vertices is not None:
                pred_positions, pred_features, importance_scores, selected_indices = model(
                    highpoly_batch, target_vertex_count=target_vertices
                )
            else:
                # Usar número de vértices do target como referência (se disponível no checkpoint)
                pred_positions, pred_features, importance_scores, selected_indices = model(highpoly_batch)
        else:
            # Modelo antigo retorna apenas 2 valores
            pred_positions, pred_features = model(highpoly_batch)
            selected_indices = None
    
    # Criar grafo do low poly
    if USE_NEW_MODEL:
        # Novo modelo já retorna número correto de vértices
        # Filtrar edge_index para vértices selecionados
        from torch_geometric.data import Data
        
        # Filtrar edge_index para usar apenas vértices selecionados
        if selected_indices is not None and highpoly_graph.edge_index.shape[1] > 0:
            # Criar mapeamento: índice original → novo índice
            n_original = highpoly_graph.edge_index.max().item() + 1
            mapping = torch.full((n_original,), -1, dtype=torch.long, device=highpoly_graph.edge_index.device)
            mapping[selected_indices] = torch.arange(len(selected_indices), device=highpoly_graph.edge_index.device)
            
            # Filtrar arestas onde ambos os vértices estão selecionados
            mask = (mapping[highpoly_graph.edge_index[0]] >= 0) & (mapping[highpoly_graph.edge_index[1]] >= 0)
            filtered_edges = highpoly_graph.edge_index[:, mask]
            
            # Remapear índices
            if filtered_edges.numel() > 0:
                filtered_edges[0] = mapping[filtered_edges[0]]
                filtered_edges[1] = mapping[filtered_edges[1]]
                filtered_edge_index = filtered_edges.cpu()
            else:
                filtered_edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            filtered_edge_index = torch.empty((2, 0), dtype=torch.long)
        
        lowpoly_graph = Data(
            pos=pred_positions.cpu(),
            x=torch.cat([pred_positions.cpu(), pred_features.cpu()], dim=1),
            edge_index=filtered_edge_index
        )
        print(f"✓ Low poly gerado com {len(lowpoly_graph.pos)} vértices (selecionados pelo modelo)")
    else:
        # Modelo antigo: criar grafo completo e reduzir se necessário
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


def process_all_highpoly(checkpoint_path, output_dir=None, target_vertices=None, device="cpu"):
    """
    Processa todos os arquivos highpoly do dataset.
    
    Args:
        checkpoint_path: Caminho para o checkpoint
        output_dir: Diretório de saída (padrão: OUTPUT_DIR)
        target_vertices: Número desejado de vértices (opcional)
        device: Dispositivo a usar
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Encontrar todos os arquivos highpoly no dataset
    highpoly_files = []
    pairs_dir = ORGANIZED_PAIRS_DIR
    
    if not pairs_dir.exists():
        print(f"❌ Erro: Diretório do dataset não encontrado: {pairs_dir}")
        return
    
    # Procurar em todos os subdiretórios pair_*
    for pair_dir in sorted(pairs_dir.glob("pair_*")):
        highpoly_file = list(pair_dir.glob("highpoly-*.obj"))
        if highpoly_file:
            highpoly_files.append(highpoly_file[0])
    
    if not highpoly_files:
        print(f"❌ Nenhum arquivo highpoly encontrado em {pairs_dir}")
        return
    
    print("=" * 70)
    print(f"Processando {len(highpoly_files)} arquivos highpoly do dataset")
    print("=" * 70)
    print()
    
    # Carregar modelo uma vez
    print(f"Carregando modelo de {checkpoint_path}...")
    model, checkpoint = load_model(checkpoint_path, device)
    print(f"✓ Modelo carregado (época {checkpoint.get('epoch', '?')})")
    print()
    
    # Processar cada arquivo
    successful = 0
    failed = 0
    
    for i, highpoly_path in enumerate(highpoly_files, 1):
        print(f"\n[{i}/{len(highpoly_files)}] Processando: {highpoly_path.name}")
        print("-" * 70)
        
        try:
            # Gerar nome do arquivo de saída
            pair_id = highpoly_path.stem.replace("highpoly-", "")
            output_path = output_dir / f"lowpoly-{pair_id}.obj"
            
            # Processar arquivo (versão simplificada sem carregar modelo novamente)
            highpoly_mesh = trimesh.load(str(highpoly_path))
            if isinstance(highpoly_mesh, trimesh.Scene):
                highpoly_mesh = highpoly_mesh.geometry[list(highpoly_mesh.geometry.keys())[0]]
            
            # Converter para grafo
            highpoly_graph = mesh_to_graph(highpoly_mesh)
            
            # Fazer predição
            highpoly_batch = highpoly_graph.to(device)
            with torch.no_grad():
                if USE_NEW_MODEL:
                    # Novo modelo retorna 4 valores
                    if target_vertices is not None:
                        pred_positions, pred_features, importance_scores, selected_indices = model(
                            highpoly_batch, target_vertex_count=target_vertices
                        )
                    else:
                        pred_positions, pred_features, importance_scores, selected_indices = model(highpoly_batch)
                else:
                    # Modelo antigo retorna apenas 2 valores
                    pred_positions, pred_features = model(highpoly_batch)
                    selected_indices = None
            
            # Criar grafo do low poly
            if USE_NEW_MODEL:
                # Filtrar edge_index para vértices selecionados
                if selected_indices is not None and highpoly_graph.edge_index.shape[1] > 0:
                    n_original = highpoly_graph.edge_index.max().item() + 1
                    mapping = torch.full((n_original,), -1, dtype=torch.long, device=highpoly_graph.edge_index.device)
                    mapping[selected_indices] = torch.arange(len(selected_indices), device=highpoly_graph.edge_index.device)
                    
                    mask = (mapping[highpoly_graph.edge_index[0]] >= 0) & (mapping[highpoly_graph.edge_index[1]] >= 0)
                    filtered_edges = highpoly_graph.edge_index[:, mask]
                    
                    if filtered_edges.numel() > 0:
                        filtered_edges[0] = mapping[filtered_edges[0]]
                        filtered_edges[1] = mapping[filtered_edges[1]]
                        filtered_edge_index = filtered_edges.cpu()
                    else:
                        filtered_edge_index = torch.empty((2, 0), dtype=torch.long)
                else:
                    filtered_edge_index = torch.empty((2, 0), dtype=torch.long)
                
                from torch_geometric.data import Data
                lowpoly_graph = Data(
                    pos=pred_positions.cpu(),
                    x=torch.cat([pred_positions.cpu(), pred_features.cpu()], dim=1),
                    edge_index=filtered_edge_index
                )
            else:
                # Modelo antigo: criar grafo completo e reduzir se necessário
                lowpoly_graph = highpoly_graph.clone()
                lowpoly_graph.pos = pred_positions.cpu()
                lowpoly_graph.x = torch.cat([pred_positions.cpu(), pred_features.cpu()], dim=1)
                
                # Reduzir número de vértices se necessário
                if target_vertices and len(lowpoly_graph.pos) > target_vertices:
                    importance = torch.norm(lowpoly_graph.pos, dim=1)
                    _, indices = torch.topk(importance, target_vertices)
                    lowpoly_graph.pos = lowpoly_graph.pos[indices]
                    lowpoly_graph.x = lowpoly_graph.x[indices]
            
            # Salvar resultado
            output_path.parent.mkdir(parents=True, exist_ok=True)
            lowpoly_mesh = graph_to_mesh(lowpoly_graph, output_path)
            
            successful += 1
            print(f"✓ Concluído: {output_path.name} ({len(lowpoly_graph.pos)} vértices)")
            
        except Exception as e:
            failed += 1
            print(f"❌ Erro ao processar {highpoly_path.name}: {e}")
            import traceback
            traceback.print_exc()
    
    print()
    print("=" * 70)
    print(f"Inferência em lote concluída!")
    print(f"  - Sucesso: {successful}/{len(highpoly_files)}")
    print(f"  - Falhas: {failed}/{len(highpoly_files)}")
    print(f"  - Resultados salvos em: {output_dir}")
    print("=" * 70)


def main():
    """
    Função principal de inferência.
    """
    parser = argparse.ArgumentParser(description="Gerar low poly mesh a partir de high poly")
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=None,
        help="Caminho para o arquivo OBJ high poly (opcional se usar --all)"
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Processar todos os arquivos highpoly do dataset"
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
    
    # Processar todos os arquivos ou um único arquivo
    if args.all:
        process_all_highpoly(
            checkpoint_path,
            args.output,
            args.target_vertices,
            device
        )
    else:
        if args.input is None:
            print("❌ Erro: --input é obrigatório ou use --all para processar todos os arquivos")
            parser.print_help()
            return
        
        # Executar inferência para um único arquivo
        predict_lowpoly(
            args.input,
            checkpoint_path,
            args.output,
            args.target_vertices,
            device
        )


if __name__ == "__main__":
    main()

