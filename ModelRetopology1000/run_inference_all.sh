#!/bin/bash
# Script para executar inferência em todos os arquivos highpoly do dataset

echo "Executando inferência para todos os arquivos highpoly do dataset..."
echo ""

# Verificar se CUDA está disponível
if command -v nvidia-smi &> /dev/null; then
    DEVICE="cuda"
    echo "CUDA detectado, usando GPU"
else
    DEVICE="cpu"
    echo "Usando CPU"
fi

# Executar inferência para todos os arquivos
python inference.py --all --device $DEVICE

echo ""
echo "Inferência concluída!"