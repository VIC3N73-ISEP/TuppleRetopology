# ModelRetopology2000

Modelo que aprende a transformar **highpoly em lowpoly** usando três entradas:

1. **Heatmap PLY** – vértices mais importantes (cores: azul = pouco importante, vermelho = importante)
2. **Highpoly** – malha de alta resolução
3. **Lowpoly** – par de referência (target)

A rede usa a importância (heatmap) para guiar a transformação.

## Preparação dos dados

Cada par deve estar numa pasta `pair_XXXX/` com:

- `highpoly-XXXX.obj`
- `lowpoly-XXXX.obj`
- `highpoly-XXXX_heatmap.ply` (gerado pelo ModuleVerticeRanker1000)

Gerar os heatmaps para uma pasta de pares (ex.: `trains_single_pair`):

```bash
cd ModuleVerticeRanker1000
python run_ranker.py -i "..\trains_single_pair"
```

Isto cria em cada `pair_XXXX/` o ficheiro `highpoly-XXXX_heatmap.ply`.

## Configuração

Em `config.py`:

- `ORGANIZED_PAIRS_DIR`: pasta dos pares (ex.: `../trains_single_pair` ou caminho absoluto)
- `MODEL_CONFIG["node_features"]`: 10 (9 geometria + 1 importância)

## Treino

```bash
cd ModelRetopology2000
python train.py
# ou continuar de um checkpoint:
python train.py --resume checkpoints/last_checkpoint.pt
```

## Inferência

```bash
python inference.py --highpoly path/to/highpoly.obj --heatmap path/to/highpoly_heatmap.ply --output path/to/lowpoly.obj
```

Opções: `--checkpoint`, `--device cuda`.
