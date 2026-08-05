"""
src/model.py

ResNet compacta para CIFAR-10, implementada do zero (sem torchvision.models).
Inspirada na família ResNet (He et al., 2015) mas dimensionada pra treinar
rápido em uma única GPU/CPU — blocos residuais com conv-bn-relu, downsampling
por stride, e um "stem" inicial mais leve que a ResNet original (que foi
desenhada pra ImageNet, imagens bem maiores que as 32x32 do CIFAR-10).

Por que implementar do zero em vez de usar torchvision.models.resnet18:
1. Pra portfólio de engenheiro de ML, mostrar que você entende os blocos
   (residual connection, batchnorm, downsampling) é mais valioso do que
   importar uma classe pronta.
2. A ResNet18 padrão do torchvision tem um stem pesado (conv 7x7 stride 2 +
   maxpool) desenhado pra imagens 224x224 — em imagens 32x32 isso destrói
   a resolução espacial cedo demais. Aqui uso um stem mais simples (conv
   3x3 stride 1), como é padrão em implementações de ResNet-para-CIFAR.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Bloco residual básico: duas convoluções 3x3 com batchnorm, e uma
    conexão de atalho (identity, ou projeção 1x1 quando o stride/canais
    mudam) somada antes da ativação final — é isso que resolve o
    vanishing gradient em redes mais profundas."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                                stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Projeção de atalho: só é necessária quando a dimensão de saída
        # muda (mais canais, ou stride > 1 reduzindo a resolução espacial).
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)  # a conexão residual
        return F.relu(out)


class ResNetCifar(nn.Module):
    """
    Stem (3x3 conv) -> 3 estágios de blocos residuais (16 -> 32 -> 64
    canais, reduzindo resolução espacial a cada estágio) -> global average
    pooling -> classificador linear.

    `num_blocks_per_stage` controla a profundidade — o padrão (2, 2, 2)
    dá uma rede de ~11 camadas convolucionais, um bom equilíbrio entre
    capacidade e tempo de treino em CPU/GPU única.
    """

    def __init__(
        self,
        num_classes: int = 10,
        in_channels: int = 3,
        num_blocks_per_stage: tuple[int, int, int] = (2, 2, 2),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        self.stage1 = self._make_stage(16, 16, num_blocks_per_stage[0], stride=1)
        self.stage2 = self._make_stage(16, 32, num_blocks_per_stage[1], stride=2)
        self.stage3 = self._make_stage(32, 64, num_blocks_per_stage[2], stride=2)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(64, num_classes)

        self._init_weights()

    @staticmethod
    def _make_stage(in_channels: int, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        layers = [ResidualBlock(in_channels, out_channels, stride)]
        for _ in range(num_blocks - 1):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def _init_weights(self):
        # Inicialização de He (kaiming), padrão para redes com ReLU —
        # sem isso, redes mais fundas tendem a treinar mal desde o início.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.global_pool(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)


def build_model(num_classes: int = 10) -> ResNetCifar:
    return ResNetCifar(num_classes=num_classes)
