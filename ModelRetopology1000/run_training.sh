#!/bin/bash
# Script auxiliar para executar o treinamento

echo "=========================================="
echo "Iniciando treinamento do modelo GNN"
echo "=========================================="
echo ""

# Verificar se os pares estão organizados
if [ ! -d "organized_pairs" ]; then
    echo "Organizando pares..."
    python organize_retopo_pairs.py
    echo ""
fi

# Executar treinamento
echo "Iniciando treinamento..."
python train.py

echo ""
echo "Treinamento concluído!"
echo "Verifique os checkpoints em: checkpoints/"

