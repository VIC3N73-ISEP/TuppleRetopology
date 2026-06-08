"""
Script rápido para criar dataset com apenas um par multiplicado 25 vezes.
Pega o par 13 e cria 25 cópias para treinamento rápido.
"""

import shutil
from pathlib import Path


def create_mass_dataset(num_pairs: int = 500):
    """
    Cria um dataset com exatamente num_pairs pares no formato esperado pelo treino:
      train_mass_pair/pair_XXXX/highpoly-XXXX.obj
      train_mass_pair/pair_XXXX/lowpoly-XXXX.obj
    Usa os pares existentes em Expai-Retopo de forma cíclica.
    """
    print("=" * 60)
    print(f"Criando Dataset massivo com {num_pairs} pares")
    print("=" * 60)
    print()

    source_dir = Path("Expai-Retopo")
    output_dir = Path("train_mass_pair")
    output_dir.mkdir(exist_ok=True)

    valid_pairs = []
    for highpoly in sorted(source_dir.glob("highpoly-*.obj")):
        source_id = highpoly.stem.split("-")[1]
        lowpoly = source_dir / f"lowpoly-{source_id}.obj"
        if lowpoly.exists():
            valid_pairs.append((highpoly, lowpoly))

    if not valid_pairs:
        print(f"Erro: nenhum par high/low valido encontrado em {source_dir}")
        return

    print(f"Criando {num_pairs} pares em {output_dir} ...")
    print(f"Pares fonte disponiveis: {len(valid_pairs)}")
    print()

    for i in range(1, num_pairs + 1):
        pair_id = f"{i:04d}"
        high_src, low_src = valid_pairs[(i - 1) % len(valid_pairs)]

        pair_dir = output_dir / f"pair_{pair_id}"
        pair_dir.mkdir(exist_ok=True)
        shutil.copy2(high_src, pair_dir / f"highpoly-{pair_id}.obj")
        shutil.copy2(low_src, pair_dir / f"lowpoly-{pair_id}.obj")

        if i % 50 == 0 or i == num_pairs:
            print(f"  Criado pair_{pair_id} ({i}/{num_pairs})")

if __name__ == "__main__":
    create_mass_dataset(500)
