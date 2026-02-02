"""
Script rápido para criar dataset com apenas um par multiplicado 25 vezes.
Pega o par 13 e cria 25 cópias para treinamento rápido.
"""

import shutil
from pathlib import Path

def create_single_pair_dataset():
    """
    Cria um dataset com apenas o par 13, multiplicado 25 vezes.
    """
    print("=" * 60)
    print("Criando Dataset com Par 13 (25x)")
    print("=" * 60)
    print()
    
    # Diretórios
    source_dir = Path("Expai-Retopo")
    output_dir = Path("trains_single_pair")
    output_dir.mkdir(exist_ok=True)
    
    # Arquivos do par 13
    highpoly_source = source_dir / "highpoly-0013.obj"
    lowpoly_source = source_dir / "lowpoly-0013.obj"
    
    # Verificar se os arquivos existem
    if not highpoly_source.exists():
        print(f"❌ Erro: {highpoly_source} não encontrado!")
        return
    
    if not lowpoly_source.exists():
        print(f"❌ Erro: {lowpoly_source} não encontrado!")
        return
    
    print(f"✓ Arquivos encontrados:")
    print(f"  - {highpoly_source}")
    print(f"  - {lowpoly_source}")
    print()
    
    # Criar 25 cópias
    num_copies = 10000
    print(f"📦 Criando {num_copies} cópias do par 13...")
    print()
    
    for i in range(1, num_copies + 1):
        # Criar pasta do par (usar numeração sequencial)
        pair_id = f"{i:04d}"
        pair_dir = output_dir / f"pair_{pair_id}"
        pair_dir.mkdir(exist_ok=True)
        
        # Copiar arquivos
        highpoly_dest = pair_dir / f"highpoly-{pair_id}.obj"
        lowpoly_dest = pair_dir / f"lowpoly-{pair_id}.obj"
        
        shutil.copy2(highpoly_source, highpoly_dest)
        shutil.copy2(lowpoly_source, lowpoly_dest)
        
        if i % 5 == 0 or i == num_copies:
            print(f"  ✓ Criado par_{pair_id} ({i}/{num_copies})")
    
    print()
    print("=" * 60)
    print(f"✅ Dataset criado com sucesso!")
    print("=" * 60)
    print()
    print(f"📁 Diretório: {output_dir}")
    print(f"📊 Total de pares: {num_copies}")
    print(f"📝 Todos os pares são cópias do par 13 original")
    print()
    print("💡 Agora você pode treinar com:")
    print("   - TRAIN_PAIRS = 20")
    print("   - VAL_PAIRS = 5")
    print("   (Total de 25 pares disponíveis)")


if __name__ == "__main__":
    create_single_pair_dataset()

