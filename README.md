# ResNet for CIFAR-10 (PyTorch)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green)
![Tests](https://img.shields.io/badge/Tests-passing-brightgreen)
![Status](https://img.shields.io/badge/Status-ready%20to%20train-yellow)

&gt; Implementação compacta de **ResNet** construída do zero em **PyTorch** para classificação de imagens no **CIFAR-10**.
&gt;
&gt; **175.258 parâmetros** | Sem `torchvision.models` | Pipeline completo de treino

---

## Highlights

- 🔧 **Construída do zero** — cada camada, skip connection e projeção implementadas manualmente
- 🧪 **Testada com sanity checks** — shape tests, gradient check e overfit em single batch validados
- 📊 **Pipeline completo** — data augmentation, label smoothing, weight decay, OneCycleLR, early stopping, checkpointing
- 🚀 **Pronta pra rodar** — basta executar `python -m src.train` para baixar o CIFAR-10 e treinar
- 📝 **Documentada pra entrevista** — cada decisão de design é justificada com argumentos técnicos

---

## Sobre este projeto

Este repositório demonstra domínio prático de:

- Arquiteturas residuais (skip connections, projection shortcuts, downsampling por stride)
- Inicialização de pesos (He / Kaiming)
- Normalização em batch (BatchNorm)
- Regularização (data augmentation, label smoothing, weight decay)
- Otimização (OneCycle Learning Rate scheduler)
- Engenharia de ML (pipeline de treino, avaliação, inferência, testes unitários)

A implementação não usa `torchvision.models` de propósito. O objetivo é mostrar entendimento profundo da arquitetura, não apenas a capacidade de chamar uma API de alto nível.

---

## ⚠️ Status honesto

O treino completo no CIFAR-10 **ainda não foi executado** — o ambiente onde o código foi escrito não tinha acesso à internet liberado para download do dataset (~170MB).

O que **foi validado** rodando o código:

| Validação | Status |
|-----------|--------|
| Shape tests para diferentes batch sizes e número de classes | ✅ |
| Gradientes fluem por todos os parâmetros (sem camada "desconectada") | ✅ |
| Overfit em single batch (sanity check: loss de ~2.3 → &lt;0.1 em 200 passos) | ✅ |
| `train_one_epoch` / `evaluate` com DataLoader real | ✅ |
| `train.py` end-to-end (checkpointing, CSV log, matriz de confusão, classification report) | ✅ |
| Script de inferência carregando checkpoint e classificando imagem | ✅ |

**O que falta você rodar:**

```bash
python -m src.train --epochs 40
