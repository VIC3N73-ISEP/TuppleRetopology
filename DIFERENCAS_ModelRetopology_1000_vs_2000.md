# Diferenças: ModelRetopology1000 vs ModelRetopology2000

Resumo direto: o **1000** usa só highpoly + lowpoly; o **2000** usa **heatmap (importância por vértice)** além do highpoly e lowpoly, e exige heatmap também na inferência.

---

## 1. Objetivo e entrada de dados

| Aspecto | ModelRetopology1000 | ModelRetopology2000 |
|--------|---------------------|----------------------|
| **Objetivo** | Aprender highpoly → lowpoly a partir da geometria | Aprender highpoly + importância (heatmap) → lowpoly |
| **Entrada de treino** | Par (highpoly.obj, lowpoly.obj) | Triplo (heatmap.ply, highpoly.obj, lowpoly.obj) |
| **Heatmap** | Não usa | **Obrigatório**: PLY com vertex colors = importância (vermelho = importante) |

---

## 2. Configuração do modelo (`config.py`)

| Parâmetro | ModelRetopology1000 | ModelRetopology2000 |
|-----------|---------------------|----------------------|
| **node_features** | 9 (posição, normal, curvatura, distância ao centro, etc.) | **10** (as mesmas 9 + **1 canal de importância do heatmap**) |
| **hidden_dim** | 256 | 256 |
| **num_layers** | 8 | 8 |
| **dropout** | 0.2 | 0.2 |
| **use_batch_norm** | True | True |
| **ORGANIZED_PAIRS_DIR** | `../trains_single_pair` (relativo ao 1000) | `parent / "trains_single_pair"` (relativo ao repo) |

O 2000 está explicitamente pensado para usar heatmap; o 1000 não.

---

## 3. Dataset

| Aspecto | ModelRetopology1000 | ModelRetopology2000 |
|--------|---------------------|----------------------|
| **Classe** | `RetopologyDataset` | `RetopologyDataset2000` |
| **Ficheiros por par** | `highpoly-{id}.obj`, `lowpoly-{id}.obj` | `highpoly-{id}.obj`, `lowpoly-{id}.obj`, **`highpoly-{id}_heatmap.ply`** |
| **Grafo highpoly** | 9 features por vértice | **10 features** (9 + importância do heatmap) |
| **Importância** | Não calculada nem usada | Lida de `_importance_from_heatmap_ply()` (canal R do PLY = score 0–1) |

No 2000, se o heatmap PLY não existir ou tiver número de vértices diferente, usa importância constante 0.5.

---

## 4. Modelo GNN

| Aspecto | ModelRetopology1000 | ModelRetopology2000 |
|--------|---------------------|----------------------|
| **Classe principal** | `RetopologyGNN` | `RetopologyGNN2000` |
| **Entrada** | Grafo com 9 features | Grafo com **10 features** (9 + importância) |
| **Arquitetura** | Encoder GCN + vertex reduction + decoder GCN + output 3D + feature decoder | Mesma ideia (encoder-decoder GCN), adaptada a 10 features |
| **Extra no 1000** | Tem também `ImprovedRetopologyGNN` (GAT + attention pooling); o train usa `RetopologyGNN` | Só `RetopologyGNN2000` |

Ou seja: mesma família encoder-decoder; 2000 recebe um canal a mais (importância).

---

## 5. Treino (`train.py`)

| Aspecto | ModelRetopology1000 | ModelRetopology2000 |
|--------|---------------------|----------------------|
| **Loss** | `_OldRetopologyLoss` (Chamfer + normal/feature, pesos configuráveis) | `RetopologyLoss` (Chamfer + MSE normais/features; pos_weight=1.2, normal_weight=0.1) |
| **Clip grad** | Pode não estar explícito em todo o fluxo | `clip_grad_norm_(model.parameters(), max_norm=1.0)` |
| **Checkpoint** | Inclui `model_config`, history, etc. | Estrutura equivalente, com `model_config` para `RetopologyGNN2000` |

O pipeline de treino (epoch, val, save) é semelhante; a diferença está na loss e no facto de o 2000 depender de batches com highpoly já com 10 features (heatmap incluído).

---

## 6. Inferência

| Aspecto | ModelRetopology1000 | ModelRetopology2000 |
|--------|---------------------|----------------------|
| **Argumentos** | `--input` (OBJ highpoly), `--checkpoint`, `--output`, `--target-vertices`, `--device` | **`--highpoly`**, **`--heatmap`** (PLY), `--checkpoint`, `--output`, `--device` |
| **Heatmap na inferência** | Não existe | **Obrigatório**: sem heatmap PLY não corre |
| **Comando 1000** | `python inference.py --input ../highpoly-0013.obj --checkpoint checkpoints/checkpoint_epoch_100.pt` | — |
| **Comando 2000** | — | `python inference.py --highpoly ../highpoly-0013.obj --heatmap ../highpoly-0013_heatmap.ply --checkpoint checkpoints/best_model.pt` |

No 2000, a importância é sempre obtida do ficheiro PLY (ou fallback 0.5 se o dataset usar isso internamente); na inferência o ficheiro heatmap é obrigatório.

---

## 7. Estrutura de ficheiros no projeto

| ModelRetopology1000 | ModelRetopology2000 |
|---------------------|----------------------|
| `config.py`, `model.py`, `dataset.py`, `train.py`, `inference.py` | Idem |
| `create_single_pair_dataset.py`, `organize_retopo_pairs.py`, `test_pipeline.py` | Não existem (ou não são necessários para o fluxo 2000) |
| `run_inference_all.sh`, `run_training.sh`, `pairs_metadata.csv/json`, `training_pairs.txt` | Não referidos no 2000 (dataset vem de `trains_single_pair` com heatmaps) |

---

## 8. Quando usar qual

- **ModelRetopology1000**: só tens pares highpoly + lowpoly (OBJ). Não queres depender de heatmaps. Inferência só com OBJ highpoly.
- **ModelRetopology2000**: tens (ou podes gerar) heatmaps PLY por highpoly (ex.: com **ModuleVerticeRanker1000**). Queres que o modelo use “importância do vértice” na decisão. Na inferência tens de ter sempre o OBJ highpoly e o PLY heatmap correspondente.

---

## 9. Resumo em uma frase

**ModelRetopology1000**: highpoly → lowpoly (só geometria).  
**ModelRetopology2000**: highpoly + heatmap (importância) → lowpoly; heatmap é obrigatório no dataset e na inferência.
