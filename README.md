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

## 📊 Documentação Técnica Detalhada

### 🧠 model.py - Arquitetura do Modelo

O arquivo `model.py` contém a implementação do modelo GNN para retopologia. A arquitetura principal é a classe `RetopologyGNN`, que implementa um encoder-decoder com mecanismo de atenção.

#### Arquitetura Encoder-Decoder

**Encoder (Processamento do High Poly):**
- **Tipo de camada**: GCN (Graph Convolutional Network) usando `GCNConv` do PyTorch Geometric
- **Número de camadas**: 6 camadas (configurável via `num_layers` em `MODEL_CONFIG`)
- **Dimensões**:
  - **Entrada**: `node_features = 9` features por vértice
  - **Hidden**: `hidden_dim = 128` dimensões ocultas
  - **Fluxo**: 9 → 128 → 128 → 128 → 128 → 128 → 128
- **Normalização**: Batch Normalization opcional (`use_batch_norm = True`)
- **Dropout**: 0.1 (10% de dropout após cada camada)
- **Ativação**: ReLU após cada camada convolucional

**Features de Entrada (9 dimensões por vértice):**
1. **Posição 3D** (3 dims): Coordenadas x, y, z do vértice
2. **Normal 3D** (3 dims): Vetor normal do vértice (nx, ny, nz)
3. **Curvatura** (1 dim): Aproximação da curvatura local calculada pela variação das normais dos vizinhos
4. **Distância ao centro** (1 dim): Distância euclidiana do vértice ao centroide do mesh
5. **Placeholder** (1 dim): Feature adicional para flexibilidade futura

**Mecanismo de Atenção (Pooling Adaptativo):**
- **Propósito**: Selecionar e ponderar vértices importantes para a retopologia
- **Arquitetura**: Rede MLP de 2 camadas
  - Camada 1: `hidden_dim (128) → hidden_dim // 2 (64)` com ReLU
  - Camada 2: `64 → 1` com Sigmoid (produz scores entre 0 e 1)
- **Uso**: Os scores de atenção são usados para ponderar as features antes da redução dimensional

**Redução de Features:**
- **Camada linear**: `hidden_dim (128) → hidden_dim // 2 (64)`
- **Propósito**: Reduzir a dimensionalidade antes do decoder, simulando a redução de vértices

**Decoder (Geração do Low Poly):**
- **Número de camadas**: 5 camadas GCN (`num_layers - 1`)
- **Dimensões**: Mantém `hidden_dim // 2 = 64` em todas as camadas
- **Fluxo**: 64 → 64 → 64 → 64 → 64
- **Normalização e Dropout**: Mesmas configurações do encoder

**Output Layers:**
1. **Output de Posições** (`output_conv`):
   - Tipo: GCNConv
   - Dimensões: `64 → 3` (coordenadas x, y, z)
   - Produz: Posições 3D dos vértices do low poly mesh

2. **Output de Features** (`feature_decoder`):
   - Tipo: MLP sequencial
   - Arquitetura:
     - Linear: `64 → 16` com ReLU
     - Dropout: 0.1
     - Linear: `16 → 6`
   - Produz: Features adicionais (normal 3D + 3 features extras)

#### Resumo das Dimensões do Modelo

- **Entrada**: Grafo com N vértices, cada um com 9 features
- **Encoder output**: N vértices com 128 features cada
- **Após atenção e redução**: N vértices com 64 features cada
- **Saída**: 
  - Posições: N vértices com 3 coordenadas (x, y, z)
  - Features: N vértices com 6 features (normal + extras)

#### Número Total de Parâmetros

O modelo tem aproximadamente:
- Encoder: ~6 camadas × (9×128 + 128×128) ≈ 100K parâmetros
- Atenção: ~128×64 + 64×1 ≈ 8K parâmetros
- Redução: ~128×64 ≈ 8K parâmetros
- Decoder: ~5 camadas × (64×64) ≈ 20K parâmetros
- Outputs: ~64×3 + 64×16 + 16×6 ≈ 1.5K parâmetros
- **Total aproximado**: ~140K parâmetros

#### Classe ImprovedRetopologyGNN

Uma versão alternativa que usa:
- **GAT (Graph Attention Network)** no encoder em vez de GCN
- **4 heads de atenção** nas camadas intermediárias
- Arquitetura similar mas com atenção explícita nas convoluções

---

### 🎓 train.py - Processo de Treinamento

O arquivo `train.py` implementa o loop de treinamento completo com loss customizada, otimizador e agendamento de learning rate.

#### Loss Function Customizada (`RetopologyLoss`)

**Problema**: High poly e low poly meshes têm números diferentes de vértices, então não podemos usar loss simples como MSE diretamente.

**Solução**: Usar métricas que funcionam com conjuntos de pontos de tamanhos diferentes.

**Componentes da Loss:**

1. **Chamfer Distance** (peso: 0.8 × `pos_weight`):
   - Para cada vértice predito, encontra o vértice mais próximo no target
   - Para cada vértice target, encontra o vértice mais próximo no predito
   - Soma as distâncias mínimas médias
   - **Fórmula**: `mean(min_dist(pred→target)) + mean(min_dist(target→pred))`
   - **Vantagem**: Funciona com números diferentes de vértices

2. **Hausdorff Distance** (peso: 0.2 × `pos_weight`):
   - Distância máxima entre os dois conjuntos de pontos
   - **Fórmula**: `max(max(min_dist(pred→target)), max(min_dist(target→pred)))`
   - **Propósito**: Garantir que a forma geral seja preservada (evita que partes importantes sejam perdidas)

3. **Feature Loss** (peso: `normal_weight = 0.5`):
   - Usa Chamfer Distance também para features (normais, etc.)
   - Calcula distância euclidiana entre features usando nearest neighbor matching
   - **Propósito**: Preservar características geométricas além da posição

**Loss Total:**
```
total_loss = pos_weight × (0.8 × chamfer + 0.2 × hausdorff) + normal_weight × feature_loss
```

**Pesos padrão:**
- `pos_weight = 1.0`
- `normal_weight = 0.5`
- `shape_weight = 0.3` (não usado diretamente, mas hausdorff serve para isso)

#### Otimizador e Scheduler

- **Otimizador**: Adam
  - **Learning rate inicial**: 0.001
  - **Parâmetros**: Usa defaults do PyTorch (beta1=0.9, beta2=0.999)

- **Scheduler**: `ReduceLROnPlateau`
  - **Modo**: 'min' (reduz LR quando loss para de diminuir)
  - **Factor**: 0.5 (reduz LR pela metade)
  - **Patience**: 10 épocas (espera 10 épocas sem melhoria antes de reduzir)
  - **Propósito**: Reduzir learning rate automaticamente quando o modelo para de melhorar

#### Gradient Clipping

- **Tipo**: Norm clipping
- **Max norm**: 1.0
- **Propósito**: Prevenir gradientes explosivos que podem causar instabilidade no treinamento

#### Configurações de Treinamento

Definidas em `TRAINING_CONFIG`:
- **Batch size**: 4 (número de pares processados por vez)
- **Número de épocas**: 100
- **Validação**: A cada 5 épocas (`val_every = 5`)
- **Checkpoints**: Salvos a cada 10 épocas (`save_every = 10`)
- **Device**: CUDA se disponível, senão CPU

#### Estrutura do Loop de Treinamento

1. **Para cada época**:
   - Treina em todos os batches do dataset de treino
   - Calcula loss média de treino
   - A cada `val_every` épocas:
     - Valida no dataset de validação
     - Atualiza learning rate via scheduler
     - Salva melhor modelo se loss de validação melhorar
   - A cada `save_every` épocas:
     - Salva checkpoint periódico
   - Salva histórico de treinamento em JSON

#### Salvamento de Checkpoints

Cada checkpoint contém:
- `epoch`: Número da época
- `model_state_dict`: Pesos do modelo
- `optimizer_state_dict`: Estado do otimizador (para continuar treinamento)
- `loss`: Loss da época
- `model_config`: Configuração do modelo (para recriar arquitetura)
- `training_config`: Configuração de treinamento

---

### 🔮 inference.py - Geração de Low Poly

O arquivo `inference.py` implementa o pipeline completo de inferência, desde o carregamento do modelo até a geração do mesh final.

#### Pipeline de Inferência

**1. Carregamento do Modelo (`load_model`):**
- Carrega checkpoint do disco
- **Compatibilidade**: Trata checkpoints antigos e novos
- **Tratamento de chaves faltantes**: Se o checkpoint tiver menos parâmetros que o modelo atual, inicializa os faltantes com valores padrão
- **Verificação de dimensões**: Verifica se as dimensões dos parâmetros são compatíveis
- Coloca modelo em modo de avaliação (`model.eval()`)

**2. Conversão Mesh → Grafo (`mesh_to_graph`):**
- Usa o mesmo método do dataset para garantir consistência
- Normaliza o mesh se configurado
- Calcula features dos vértices (posição, normal, curvatura, etc.)
- Cria arestas a partir das faces do mesh

**3. Predição (`predict_lowpoly`):**
- Converte high poly mesh para grafo
- Executa forward pass do modelo (sem gradientes)
- Obtém posições preditas e features

**4. Desnormalização:**
- **Problema**: O modelo foi treinado com dados normalizados
- **Solução**: Tenta encontrar o low poly original correspondente para usar sua escala real
- **Fallback**: Se não encontrar, estima escala como 85% da escala do high poly
- **Fórmula**: `pred_positions = pred_positions * scale + center`

**5. Redução de Vértices (Opcional):**
- Se `target_vertices` for especificado e o número de vértices preditos for maior:
  - Calcula importância de cada vértice (norma da posição)
  - Seleciona os top-k vértices mais importantes
  - Remove os demais

**6. Conversão Grafo → Mesh (`graph_to_mesh`):**
- **Estratégia 1**: Se faces originais existirem e número de vértices corresponder, usa as faces originais
- **Estratégia 2**: Se não, tenta reconstruir faces usando Delaunay triangulation (projeta em 2D via PCA)
- **Estratégia 3**: Fallback para convex hull
- **Estratégia 4**: Último recurso: salva apenas como point cloud (sem faces)

**7. Validação e Reparo do Mesh:**
- Preenche buracos se necessário
- Remove faces duplicadas
- Remove vértices não referenciados
- Valida se o mesh é válido antes de salvar

#### Processamento em Lote (`process_all_highpoly`)

- Encontra todos os arquivos high poly no dataset
- Processa cada um sequencialmente
- Carrega modelo uma vez e reutiliza para todos
- Gera estatísticas de sucesso/falhas

#### Tratamento de Erros

- **Meshes inválidos**: Tenta reparar automaticamente
- **Faces faltantes**: Reconstrói usando triangulação
- **Topologia incompatível**: Ajusta faces ou remove vértices inválidos
- **Erros de carregamento**: Continua processamento e reporta erros

#### Argumentos da Linha de Comando

- `--input, -i`: Arquivo high poly único
- `--all, -a`: Processa todos os high poly do dataset
- `--checkpoint, -c`: Caminho do checkpoint (padrão: `best_model.pt` ou `last_checkpoint.pt`)
- `--output, -o`: Diretório ou arquivo de saída
- `--target-vertices, -v`: Número desejado de vértices (redução adicional)
- `--device`: CPU ou CUDA

---

## 📈 Métricas de Loss (Resumo)

A loss combina:
- **Posição**: Chamfer distance (80%) + Hausdorff distance (20%) entre vértices preditos e alvo
- **Features**: Chamfer distance para normais e outras características
- **Preservação de forma**: Hausdorff distance garante que a forma geral seja preservada

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

## 📋 Tabela de Referência Rápida

### Especificações do Modelo

| Componente | Valor | Descrição |
|------------|-------|-----------|
| **Features de entrada** | 9 | Posição (3) + Normal (3) + Curvatura (1) + Dist. centro (1) + Placeholder (1) |
| **Dimensão oculta** | 128 | Dimensão das features após encoder |
| **Número de camadas** | 6 | Camadas GCN no encoder |
| **Dimensão após redução** | 64 | Dimensão antes do decoder |
| **Features de saída** | 3 + 6 | Posições (3) + Features extras (6) |
| **Dropout** | 0.1 | Taxa de dropout (10%) |
| **Batch Normalization** | Sim | Usado em todas as camadas |

### Configurações de Treinamento

| Parâmetro | Valor | Descrição |
|-----------|-------|-----------|
| **Batch size** | 4 | Pares processados por batch |
| **Learning rate** | 0.001 | Taxa de aprendizado inicial |
| **Épocas** | 100 | Número total de épocas |
| **Validação** | A cada 5 épocas | Frequência de validação |
| **Checkpoints** | A cada 10 épocas | Frequência de salvamento |
| **Gradient clipping** | 1.0 | Norma máxima dos gradientes |
| **Scheduler patience** | 10 | Épocas sem melhoria antes de reduzir LR |

### Pesos da Loss Function

| Componente | Peso | Descrição |
|------------|------|-----------|
| **Posição (Chamfer)** | 0.8 × 1.0 = 0.8 | Distância entre vértices preditos e alvo |
| **Posição (Hausdorff)** | 0.2 × 1.0 = 0.2 | Distância máxima (preservação de forma) |
| **Features** | 0.5 | Normais e características geométricas |
| **Normal Consistency** | 0.3 | Consistência de normais calculadas da topologia |

### Fluxo de Dados

```
High Poly Mesh (N vértices)
    ↓
[Conversão para Grafo]
    ↓
Grafo: N nós × 9 features
    ↓
[Encoder GCN: 6 camadas]
    ↓
N nós × 128 features
    ↓
[Atenção + Redução]
    ↓
N nós × 64 features
    ↓
[Decoder GCN: 5 camadas]
    ↓
N nós × 64 features
    ↓
[Output Layers]
    ↓
Posições: N × 3 | Features: N × 6
    ↓
[Conversão para Mesh]
    ↓
Low Poly Mesh (N vértices)
```

## ⚠️ Limitação Atual: Simplificação vs Retopologia

**IMPORTANTE**: O sistema atual faz **simplificação geométrica**, não retopologia completa.

### O Que Falta:

- ❌ **Predição de conectividade**: O modelo só prevê posições de vértices, não arestas/faces
- ❌ **Loss de topologia**: A loss não penaliza conectividade/topologia
- ❌ **Reconstrução fraca**: Usa heurísticas básicas (PCA+Delaunay) para reconstruir faces

### 3 Caminhos para Retopologia Real:

#### **Caminho 1 — Aprender Conectividade** (Difícil, mas é retopo de verdade)

**O que fazer:**
- Output do modelo inclui arestas/faces (edge prediction + face classification)
- Similar ao que já foi explorado com NMS em outros trabalhos
- Loss passa a penalizar não só geometria (Chamfer) mas também conectividade/topologia

**Implementação:**
- Adicionar `edge_prediction_head`: MLP que prediz probabilidade de aresta entre vértices
- Adicionar `face_classification_head`: Classifica triângulos válidos
- Loss de topologia: penaliza arestas/faces preditas incorretas
- NMS (Non-Maximum Suppression) para filtrar faces sobrepostas

#### **Caminho 2 — Melhorar Heurísticas de Conectividade**

**Em vez de PCA+Delaunay, usar:**

1. **Poisson Surface Reconstruction** (se tiver normais boas)
   - Melhor que Delaunay para superfícies suaves
   - Requer normais precisas

2. **Ball-Pivoting Algorithm**
   - Reconstrói mesh a partir de point cloud
   - Funciona bem com densidade uniforme

3. **Remeshing baseado em SDF/Occupancy**
   - Pipeline neural implicit → mesh
   - Mais robusto para formas complexas

4. **Quadrangulation** (se precisar de quads)
   - Quadriflow ou métodos similares
   - Aplicado depois da triangulação

#### **Caminho 3 — Reformular Objetivo: "Simplificação Geométrica"**

**Se o objetivo real for só reduzir mantendo forma:**

- Assumir "tri mesh simplificada" como output
- Focar em:
  - ✅ **Melhor critério de downsampling**: Curvatura/erro QEM, etc.
  - ✅ **Loss com amostragem de superfície**: Não só vértices
  - ✅ **Normal consistency**: Garantir que normais sejam consistentes (IMPLEMENTADO)

**Normal Consistency (✅ Implementado):**
- Calcula normais a partir das posições preditas usando a topologia do grafo (arestas)
- Compara com normais do target usando Chamfer distance
- Garante que a superfície gerada tenha orientação correta
- **Peso padrão**: 0.3 (configurável via `normal_consistency_weight`)
- **Método**: Para cada vértice, calcula normal como média dos cross products entre vetores para vizinhos consecutivos

---

## 🎯 Próximos Passos

### Curto Prazo (Melhorias Rápidas):
- [x] ✅ Implementar Normal Consistency Loss (Caminho 3) - **CONCLUÍDO**
- [ ] Implementar Poisson Surface Reconstruction (Caminho 2)
- [ ] Adicionar surface sampling loss (Caminho 3)
- [ ] Melhorar critério de downsampling com QEM (Caminho 3)

### Médio Prazo (Retopologia Real):
- [ ] Adicionar edge prediction head ao modelo (Caminho 1)
- [ ] Adicionar face classification head ao modelo (Caminho 1)
- [ ] Implementar topology consistency loss (Caminho 1)
- [ ] Implementar NMS para faces (Caminho 1)

### Longo Prazo:
- Experimentar diferentes arquiteturas GNN (GAT, GraphSAGE)
- Adicionar mais features aos vértices (curvatura gaussiana, mean curvature)
- Implementar métricas de avaliação mais sofisticadas (visual quality metrics)
- Adicionar visualização durante o treinamento
- Implementar early stopping baseado em métricas de validação
- Adicionar suporte para meshes com texturas/UVs
- Suporte para quads (Quadriflow)

