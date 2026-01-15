"""
Script para organizar e emparelhar arquivos highpoly e lowpoly
para treinamento de IA de retopologia.
"""

import os
import json
import shutil
from pathlib import Path
from typing import List, Dict, Tuple

def find_pairs(directory: str) -> List[Dict[str, str]]:
    """
    Encontra e emparelha arquivos highpoly e lowpoly.
    
    Args:
        directory: Caminho da pasta com os arquivos
        
    Returns:
        Lista de dicionários com informações dos pares
    """
    dir_path = Path(directory)
    pairs = []
    
    # Encontra todos os arquivos highpoly e lowpoly
    highpoly_files = sorted(dir_path.glob("highpoly-*.obj"))
    lowpoly_files = sorted(dir_path.glob("lowpoly-*.obj"))
    
    # Cria um dicionário para acesso rápido aos lowpoly
    lowpoly_dict = {}
    for lowpoly in lowpoly_files:
        # Extrai o número do arquivo (ex: "0001" de "lowpoly-0001.obj")
        number = lowpoly.stem.split("-")[1]
        lowpoly_dict[number] = lowpoly
    
    # Emparelha os arquivos
    for highpoly in highpoly_files:
        number = highpoly.stem.split("-")[1]
        
        if number in lowpoly_dict:
            pairs.append({
                "pair_id": number,
                "highpoly": str(highpoly),
                "lowpoly": str(lowpoly_dict[number]),
                "highpoly_name": highpoly.name,
                "lowpoly_name": lowpoly_dict[number].name
            })
        else:
            print(f"⚠️  Aviso: highpoly-{number}.obj não tem par lowpoly correspondente")
    
    return pairs

def create_organized_structure(pairs: List[Dict[str, str]], output_dir: str = "organized_pairs"):
    """
    Cria uma estrutura organizada com os pares.
    
    Args:
        pairs: Lista de pares encontrados
        output_dir: Diretório de saída
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Cria subpastas para cada par
    for pair in pairs:
        pair_dir = output_path / f"pair_{pair['pair_id']}"
        pair_dir.mkdir(exist_ok=True)
        
        # Copia os arquivos para a pasta do par
        shutil.copy2(pair['highpoly'], pair_dir / pair['highpoly_name'])
        shutil.copy2(pair['lowpoly'], pair_dir / pair['lowpoly_name'])
        
        print(f"✓ Par {pair['pair_id']} organizado em {pair_dir}")
    
    return output_path

def create_metadata_file(pairs: List[Dict[str, str]], output_file: str = "pairs_metadata.json"):
    """
    Cria um arquivo JSON com metadados de todos os pares.
    
    Args:
        pairs: Lista de pares encontrados
        output_file: Nome do arquivo de saída
    """
    metadata = {
        "total_pairs": len(pairs),
        "pairs": pairs
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Metadados salvos em {output_file}")
    return output_file

def create_training_list(pairs: List[Dict[str, str]], output_file: str = "training_pairs.txt"):
    """
    Cria um arquivo de texto simples com lista de pares para treinamento.
    
    Args:
        pairs: Lista de pares encontrados
        output_file: Nome do arquivo de saída
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Lista de pares para treinamento de IA de retopologia\n")
        f.write("# Formato: pair_id|highpoly_path|lowpoly_path\n\n")
        
        for pair in pairs:
            f.write(f"{pair['pair_id']}|{pair['highpoly']}|{pair['lowpoly']}\n")
    
    print(f"✓ Lista de treinamento salva em {output_file}")
    return output_file

def create_csv_metadata(pairs: List[Dict[str, str]], output_file: str = "pairs_metadata.csv"):
    """
    Cria um arquivo CSV com metadados dos pares.
    
    Args:
        pairs: Lista de pares encontrados
        output_file: Nome do arquivo de saída
    """
    import csv
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Pair ID', 'Highpoly File', 'Lowpoly File', 'Highpoly Path', 'Lowpoly Path'])
        
        for pair in pairs:
            writer.writerow([
                pair['pair_id'],
                pair['highpoly_name'],
                pair['lowpoly_name'],
                pair['highpoly'],
                pair['lowpoly']
            ])
    
    print(f"✓ CSV de metadados salvo em {output_file}")
    return output_file

def main():
    """Função principal do script."""
    print("=" * 60)
    print("Organizador de Pares para Treinamento de IA de Retopologia")
    print("=" * 60)
    print()
    
    # Define o diretório de entrada
    input_dir = "Expai-Retopo"
    
    if not os.path.exists(input_dir):
        print(f"❌ Erro: Pasta '{input_dir}' não encontrada!")
        return
    
    print(f"📁 Procurando pares em: {input_dir}")
    print()
    
    # Encontra os pares
    pairs = find_pairs(input_dir)
    
    if not pairs:
        print("❌ Nenhum par encontrado!")
        return
    
    print(f"\n✅ Encontrados {len(pairs)} pares válidos!")
    print()
    
    # Cria estrutura organizada
    print("📦 Criando estrutura organizada...")
    organized_dir = create_organized_structure(pairs)
    print()
    
    # Cria arquivo de metadados JSON
    print("📄 Criando arquivo de metadados JSON...")
    create_metadata_file(pairs)
    print()
    
    # Cria lista de treinamento
    print("📋 Criando lista de treinamento...")
    create_training_list(pairs)
    print()
    
    # Cria CSV de metadados
    print("📊 Criando CSV de metadados...")
    create_csv_metadata(pairs)
    print()
    
    print("=" * 60)
    print("✅ Organização concluída com sucesso!")
    print("=" * 60)
    print()
    print("Estrutura criada:")
    print(f"  - {organized_dir}/ (pastas organizadas por par)")
    print("  - pairs_metadata.json (metadados em JSON)")
    print("  - training_pairs.txt (lista simples para treinamento)")
    print("  - pairs_metadata.csv (metadados em CSV)")
    print()

if __name__ == "__main__":
    main()

