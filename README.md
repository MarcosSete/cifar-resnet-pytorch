# ResNet for CIFAR-10 — PyTorch

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green)
![Tests](https://img.shields.io/badge/Tests-passing-brightgreen)
![Status](https://img.shields.io/badge/Status-training%20benchmark%20pending-yellow)

> Implementação de uma arquitetura **ResNet em PyTorch**, construída manualmente para classificação de imagens no **CIFAR-10**.
>
> **175.258 parâmetros** · Sem `torchvision.models` · Pipeline de treinamento, avaliação e inferência

---

## Objetivo

Este projeto foi desenvolvido como uma demonstração prática de **Deep Learning com PyTorch**, com foco não apenas na arquitetura do modelo, mas também nas decisões e componentes que fazem parte de um pipeline de treinamento real.

A implementação procura demonstrar capacidade de:

- projetar e implementar uma arquitetura residual sem depender de modelos prontos;
- trabalhar diretamente com tensores, convoluções, normalização e conexões residuais;
- estruturar um pipeline de treinamento reproduzível;
- aplicar técnicas de regularização e otimização;
- criar rotinas de avaliação, checkpointing e inferência;
- validar componentes do modelo com testes e sanity checks.

O uso de `torchvision.models` foi evitado intencionalmente para tornar explícita a implementação dos principais componentes da arquitetura.

---

## Implementação

A ResNet implementada inclui:

- **Residual blocks** com skip connections;
- **Projection shortcuts** para mudanças de dimensão;
- Downsampling por **stride**;
- **Batch Normalization**;
- Inicialização **Kaiming/He**;
- Global Average Pooling;
- Classificação para as 10 classes do CIFAR-10.

O modelo possui **175.258 parâmetros treináveis**.

### Pipeline de treinamento

O projeto também implementa componentes normalmente encontrados em pipelines de treinamento de modelos de visão:

- CIFAR-10 via `torchvision`;
- Data augmentation;
- Label smoothing;
- Weight decay;
- OneCycleLR;
- Early stopping;
- Checkpointing;
- Métricas de avaliação;
- Classification report;
- Matriz de confusão;
- Inferência a partir de checkpoint.

---

## Validação

Antes de executar um treinamento completo, a implementação foi validada em diferentes níveis para reduzir o risco de erros silenciosos no modelo e no pipeline:

| Validação | Status |
|---|---|
| Shape tests para diferentes batch sizes e número de classes | ✅ |
| Verificação do fluxo de gradientes por todos os parâmetros | ✅ |
| Overfit em single batch como sanity check | ✅ |
| `train_one_epoch` / `evaluate` com DataLoader | ✅ |
| Pipeline de treinamento e componentes de avaliação | ✅ |
| Checkpoint loading e inferência | ✅ |

No sanity check de overfit em um único batch, a loss foi reduzida de aproximadamente **2.3 para < 0.1 em 200 passos**, indicando que o modelo consegue aprender o sinal presente nos dados e que o fluxo de forward/backward está funcional.

---

## Status do treinamento completo

O benchmark completo no CIFAR-10 ainda não faz parte dos resultados publicados neste repositório.

Isso é uma limitação de **ambiente de execução**, e não uma limitação do pipeline implementado: o desenvolvimento e a validação do projeto ocorreram em um ambiente sem uma GPU adequada para executar o treinamento completo de forma eficiente, além de restrições de acesso à internet durante a execução.

Por esse motivo, não são apresentados neste README números de acurácia que não tenham sido efetivamente medidos.

O projeto está estruturado para que o treinamento completo possa ser executado posteriormente em um ambiente com GPU e acesso ao dataset:

```bash
python -m src.train --epochs 40
```

Essa separação entre **validação de implementação** e **benchmark experimental** é intencional: os resultados publicados devem refletir somente experimentos realmente executados.

---

## Estrutura

```text
cifar-resnet-pytorch/
├── src/
│   ├── model.py       # Arquitetura ResNet
│   ├── data.py        # Dataset e DataLoaders
│   ├── train.py       # Pipeline de treinamento
│   └── ...
├── tests/             # Testes e sanity checks
├── requirements.txt
└── README.md
```

---

## Execução

Instale as dependências:

```bash
pip install -r requirements.txt
```

Execute o treinamento:

```bash
python -m src.train --epochs 40
```

O pipeline utiliza o CIFAR-10 disponibilizado pelo `torchvision`, realizando o download automaticamente quando o dataset ainda não está disponível localmente.

---

## Tecnologias

- **Python**
- **PyTorch**
- **Torchvision**
- **NumPy**
- **scikit-learn**
- **Matplotlib**
- **Pytest**

---

## O que este projeto demonstra

Mais do que reproduzir uma arquitetura conhecida, este projeto demonstra experiência prática com o ciclo de desenvolvimento de um modelo de Deep Learning:

**Arquitetura → dados → treinamento → otimização → validação → avaliação → checkpoint → inferência**

O foco está em compreender e implementar os componentes do pipeline, em vez de apenas utilizar uma implementação pronta de ResNet.

---

## Licença

MIT
