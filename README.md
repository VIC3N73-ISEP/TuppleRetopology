# Sistema de Retopologia com GNN

Sistema completo de treinamento e inferência de IA para retopologia usando Graph Neural Networks (GNN). O modelo aprende a converter meshes high poly em low poly mantendo a forma e características importantes.

## 📋 Estrutura do Projeto

```
.
├── config.py              # Configurações do projeto (X e Y pares)
├── model.py               # Modelos GNN (RetopologyGNN, ImprovedRetopologyGNN)
├── dataset.py             # Dataset loader e conversão mesh->grafo
├── train.py               # Script de treinamento
├── inference.py           # Script de inferência
├── requirements.txt       # Dependências Python
├── organize_retopo_pairs.py # Script auxiliar para organizar pares
├── Expai-Retopo/         # Pasta com os arquivos OBJ originais
├── organized_pairs/      # Pares organizados (criado pelo organize_retopo_pairs.py)
├── checkpoints/          # Checkpoints do modelo (criado durante treino)
└── output/               # Resultados da inferência
```

## 🚀 Instalação

1. Instalar dependências:

```bash
pip install -r requirements.txt
```

2. Organizar os pares de dados (se ainda não foi feito):

```bash
python organize_retopo_pairs.py
```

## ⚙️ Configuração

Edite `config.py` para ajustar:

- **TRAIN_PAIRS**: Número de pares para treino (X)
- **VAL_PAIRS**: Número de pares para validação/avaliação (Y)
- **MODEL_CONFIG**: Configurações do modelo GNN
- **TRAINING_CONFIG**: Configurações de treinamento (batch size, learning rate, etc.)

Exemplo:
```python
TRAIN_PAIRS = 20  # X pares para treino
VAL_PAIRS = 5     # Y pares para validação
```

## 🎓 Treinamento

Para treinar o modelo:

```bash
python train.py
```

O script irá:
1. Carregar X pares para treino e Y pares para validação
2. Treinar o modelo GNN
3. Salvar checkpoints periodicamente
4. Salvar o melhor modelo em `checkpoints/best_model.pt`

### Checkpoints

- `checkpoints/best_model.pt`: Melhor modelo (menor loss de validação)
- `checkpoints/last_checkpoint.pt`: Último checkpoint
- `checkpoints/checkpoint_epoch_N.pt`: Checkpoints periódicos
- `checkpoints/training_history.json`: Histórico de treinamento

## 🔮 Inferência

Para gerar um low poly mesh a partir de um high poly:

```bash
python inference.py --input caminho/para/highpoly.obj --checkpoint checkpoints/best_model.pt --output resultado.obj
```

Opções:
- `--input, -i`: Caminho para o arquivo OBJ high poly (obrigatório)
- `--checkpoint, -c`: Caminho para o checkpoint (padrão: `checkpoints/best_model.pt`)
- `--output, -o`: Caminho para salvar o resultado (padrão: `output/lowpoly_<nome>.obj`)
- `--target-vertices, -v`: Número desejado de vértices (opcional)
- `--device`: Dispositivo a usar (`cpu` ou `cuda`)

Exemplo:
```bash
python inference.py -i Expai-Retopo/highpoly-0001.obj -o resultado.obj
```

## 📊 Modelo GNN

O modelo usa uma arquitetura encoder-decoder:

1. **Encoder**: Processa o grafo do high poly mesh
2. **Redução de vértices**: Seleciona vértices importantes
3. **Decoder**: Gera o grafo do low poly mesh
4. **Output**: Posições 3D e features (normais, etc.)

### Features dos Vértices

Cada vértice é representado por:
- Posição 3D (x, y, z)
- Normal 3D (nx, ny, nz)
- Curvatura aproximada
- Distância ao centro
- Outras features

## 📈 Métricas de Loss

A loss combina:
- **Posição**: Chamfer distance entre vértices preditos e alvo
- **Features**: Normais e outras características
- **Preservação de forma**: Manter características importantes do mesh original

## 🔧 Troubleshooting

### Erro ao carregar meshes
- Verifique se os arquivos OBJ estão no formato correto
- Alguns meshes podem precisar de processamento prévio

### Memória insuficiente
- Reduza `max_vertices` em `config.py`
- Reduza `batch_size` em `TRAINING_CONFIG`
- Use CPU se CUDA não estiver disponível

### Modelo não converge
- Ajuste `learning_rate` em `TRAINING_CONFIG`
- Aumente `num_epochs`
- Verifique se os dados estão corretos

## 📝 Notas

- O modelo funciona melhor com meshes normalizados
- A qualidade do resultado depende da qualidade dos dados de treino
- Para melhores resultados, use mais pares de treino (aumente X)
- O modelo pode precisar de ajustes finos para diferentes tipos de meshes

## 🎯 Próximos Passos

- Experimentar diferentes arquiteturas GNN
- Adicionar mais features aos vértices
- Implementar métricas de avaliação mais sofisticadas
- Adicionar visualização durante o treinamento

