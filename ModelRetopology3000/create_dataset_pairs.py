"""
Cria N pares em trains_single_pair/ a partir de:
  - high: highpoly-0013.obj (raiz do repositório ou Expai-Retopo)
  - low:  teste_lowpoli.obj (+ teste_lowpoli.mtl se existir)

Cada par: pair_XXXX/highpoly-XXXX.obj, pair_XXXX/lowpoly-XXXX.obj
"""

import argparse
import shutil
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent


def resolve_highpoly() -> Path:
    candidates = [
        REPO_ROOT / "highpoly-0013.obj",
        REPO_ROOT / "Expai-Retopo" / "highpoly-0013.obj",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "highpoly-0013.obj não encontrado. Coloca-o na raiz do repo ou em Expai-Retopo/."
    )


def main():
    parser = argparse.ArgumentParser(description="Gerar pares high/low para trains_single_pair")
    parser.add_argument(
        "--num",
        type=int,
        default=4400,
        help="Número de pares a criar (default: 4400)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Pasta de saída (default: ../trains_single_pair)",
    )
    args = parser.parse_args()

    high_src = resolve_highpoly()
    low_src = SCRIPT_DIR / "teste_lowpoli.obj"
    mtl_src = SCRIPT_DIR / "teste_lowpoli.mtl"

    if not low_src.exists():
        raise FileNotFoundError(f"Não encontrado: {low_src}")

    out_root = args.output if args.output is not None else REPO_ROOT / "trains_single_pair"
    out_root.mkdir(parents=True, exist_ok=True)

    num = max(1, int(args.num))
    print(f"High:  {high_src}")
    print(f"Low:   {low_src}")
    print(f"Saída: {out_root}")
    print(f"Pares: {num} (pair_0001 … pair_{num:04d})")
    print()

    for i in range(1, num + 1):
        pair_id = f"{i:04d}"
        pair_dir = out_root / f"pair_{pair_id}"
        pair_dir.mkdir(exist_ok=True)
        shutil.copy2(high_src, pair_dir / f"highpoly-{pair_id}.obj")
        shutil.copy2(low_src, pair_dir / f"lowpoly-{pair_id}.obj")
        if mtl_src.exists():
            shutil.copy2(mtl_src, pair_dir / "teste_lowpoli.mtl")

        if i % 500 == 0 or i == num:
            print(f"  … {i}/{num}")

    print()
    print(f"Concluído: {num} pares em {out_root}")


if __name__ == "__main__":
    main()
