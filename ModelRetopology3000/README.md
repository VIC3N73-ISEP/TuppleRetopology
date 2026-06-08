# ModelRetopology3000

Modelo independente que aprende um **ratio de Decimate** (estilo Blender: 0 = pouca redução, 1 = redução máxima) a partir do high poly.
A saída final é **sempre decimate-only** (quadric decimation via trimesh), sem refinamento geométrico GNN.
Na configuração atual, o modelo prevê `r_final` e a inferência aplica `r_apply = 1 - r_final` no Decimate.

Não altera `ModelRetopology1000`, `ModelRetopology2000` nem outros módulos do repositório.

## Dados

- Pares em `../trains_single_pair/pair_XXXX/` com `highpoly-XXXX.obj` e `lowpoly-XXXX.obj` (mesma convenção que os outros modelos).
- O rótulo `r_gt` é derivado das faces: `1 - F_low / F_high` (clip [0,1]), ver `config.py`.

## Mapeamento ratio → faces (treino / inferência)

```text
target_faces = max(4, round(F_high * (1 - r)))
```

`target_faces` é passado a `trimesh.Trimesh.simplify_quadric_decimation`. A paridade exata com o modifier Decimate do Blender não é garantida (usa-se quadric do trimesh).

## Semântica de ratio (treino vs inferência)

- **Treino** aprende `r_final` a partir do par high/low.
- **Inferência** aplica no Decimate: `r_apply = 1 - r_final`.

Exemplo:
- se o modelo prevê `r_final = 0.9621`
- então o Blender Decimate recebe `r_apply = 0.0379`.

## Treino

```bash
cd ModelRetopology3000
python train.py
```

Checkpoints em `checkpoints/`, melhor modelo em `checkpoints/best_model.pt`.

## Inferência

**Ratio automático** (prevê `r`, aplica decimate e exporta):

```bash
python inference.py --input path/to/mesh.obj --checkpoint checkpoints/best_model.pt --output out.obj
```

**Ratio manual** (0–1):

```bash
python inference.py --input path/to/mesh.obj --ratio 0.5 --checkpoint checkpoints/best_model.pt -o out.obj
```

## Dependências

Ver `requirements.txt`.

## Loss

- `L = w_ratio * MSE(r_hat, r_gt)` (pesos em `config.py` → `LOSS_CONFIG`).
