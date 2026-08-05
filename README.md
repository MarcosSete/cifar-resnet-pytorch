# ResNet compacta para CIFAR-10 (PyTorch, do zero)

Classificador de imagens no benchmark CIFAR-10 usando uma ResNet
implementada do zero (sem `torchvision.models`) — blocos residuais,
batchnorm, inicialização de He, data augmentation, LR scheduling
(OneCycle), early stopping e checkpointing do melhor modelo.

**175.258 parâmetros.**

## ⚠️ Status honesto deste repositório

O treino completo **não foi executado** neste ambiente — o CIFAR-10 exige
download (~170MB) e o ambiente onde este código foi escrito não tinha
acesso à internet liberado para isso.

O que **foi** validado, de verdade, rodando o código:
- ✅ Arquitetura: testes de shape para diferentes batch sizes e número de classes
- ✅ Gradientes fluem por todos os parâmetros (sem camada "desconectada")
- ✅ **Overfit em um único batch**: o sanity check mais importante antes de
  rodar um treino longo — o modelo consegue levar a loss de ~2.3 (chute
  aleatório) para <0.1 em 200 passos sobre 8 exemplos sintéticos
- ✅ `train_one_epoch` / `evaluate` rodando com um DataLoader real
- ✅ `train.py` de ponta a ponta (checkpointing, log CSV, matriz de
  confusão, classification report) — testado com dados sintéticos no
  lugar do CIFAR-10 real
- ✅ Script de inferência (`scripts/predict.py`) carregando checkpoint e
  classificando uma imagem

O que falta **você** rodar e preencher abaixo com números reais:

```bash
python -m src.train --epochs 40
```

Em uma GPU comum, isso deve levar entre 15-30 min e chegar a **~92-94%**
de acurácia no teste (essa faixa é o que arquiteturas ResNet compactas
similares tipicamente atingem em CIFAR-10 na literatura — não é uma
promessa, é o benchmark que você deve conseguir bater ou explicar se
ficar longe disso).

### Resultados (preencha depois de rodar)

| Métrica | Valor |
|---|---|
| Acurácia no teste | _rode e preencha_ |
| Épocas até convergência | _rode e preencha_ |
| Tempo de treino | _rode e preencha_ |

## Instalação

```bash
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Se você tem GPU NVIDIA, instale o PyTorch com suporte CUDA seguindo as
instruções em https://pytorch.org/get-started/locally/ em vez do
`torch` puro do requirements.txt (que puxa a versão CPU-only por padrão
em alguns ambientes).

## Uso

```bash
# Treina (baixa o CIFAR-10 automaticamente na primeira vez)
python -m src.train --epochs 40 --batch-size 128 --lr 0.1

# Classifica uma imagem com o checkpoint treinado
python -m scripts.predict --checkpoint results/best_model.pt --image caminho/imagem.png
```

## Testes

```bash
pip install pytest
PYTHONPATH=. pytest tests/ -v
```

Os testes não dependem do CIFAR-10 — usam tensores sintéticos, então
rodam em segundos e servem como CI (adicione um workflow do GitHub
Actions rodando isso a cada push, se quiser deixar o repo ainda mais
convincente pra quem for avaliar).

## Decisões de design (pra defender em entrevista)

- **ResNet do zero, não `torchvision.models`**: mostra entendimento da
  arquitetura (conexão residual, downsampling por stride, projeção de
  atalho quando a dimensão muda).
- **Stem leve (conv 3x3, sem maxpool inicial)**: a ResNet original foi
  desenhada pra ImageNet (224x224); em CIFAR-10 (32x32) um stem pesado
  destrói resolução espacial cedo demais — essa é uma adaptação padrão
  na literatura de "ResNet para CIFAR".
- **Validação separada do teste**: sem isso, qualquer decisão de model
  selection (early stopping, escolha de hyperparâmetro) "vaza"
  informação do conjunto de teste, inflando a métrica reportada de forma
  não realista.
- **Label smoothing + weight decay + data augmentation**: três técnicas
  de regularização independentes — em um dataset do tamanho do CIFAR-10,
  overfitting é o principal risco, não underfitting.
- **OneCycleLR**: converge mais rápido que LR fixo em orçamentos de
  poucas épocas (relevante se você for treinar em CPU/GPU limitada).

## Estrutura

```
src/
├── model.py      # ResNetCifar, ResidualBlock
├── data.py        # CIFAR-10 + augmentation + split train/val/test
├── engine.py      # train_one_epoch, evaluate
├── metrics.py     # matriz de confusão, classification report
├── utils.py       # seed, checkpointing, AverageMeter
└── train.py       # script principal (CLI)
scripts/
└── predict.py      # inferência em uma imagem
tests/
└── test_model.py    # shape tests + overfit-single-batch
```

## Próximos passos (se quiser aprofundar depois)

1. Adicionar `torch.cuda.amp` (mixed precision) pra treinar mais rápido em GPU.
2. Exportar pra ONNX/TorchScript e servir com FastAPI (mesmo padrão dos
   outros projetos do portfólio).
3. Comparar contra uma baseline com `torchvision.models.resnet18`
   pré-treinada (transfer learning) — interessante pra discutir em
   entrevista o trade-off "treinar do zero vs. fine-tuning".
4. Grad-CAM para visualizar o que o modelo está "olhando" nas predições.
