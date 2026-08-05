"""
src/data.py

Carrega o CIFAR-10 via torchvision (baixa automaticamente na primeira
execução — precisa de internet). Aplica data augmentation padrão pra esse
benchmark (random crop com padding + flip horizontal), que é o que
diferencia "treinei um modelo" de "treinei um modelo que generaliza".
"""

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# Média e desvio padrão por canal do CIFAR-10 (valores padrão da literatura,
# calculados sobre o dataset de treino inteiro) — normalizar com esses
# valores específicos do dataset converge mais rápido do que usar
# valores genéricos tipo 0.5/0.5.
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def get_transforms(train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


def get_dataloaders(
    data_dir: str = "./data",
    batch_size: int = 128,
    val_fraction: float = 0.1,
    num_workers: int = 2,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Devolve (train_loader, val_loader, test_loader).

    O CIFAR-10 só vem com train/test oficiais — separamos uma fatia do
    treino pra validação, pra poder fazer early stopping e model selection
    sem "vazar" informação do conjunto de teste.
    """
    train_full = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=get_transforms(train=True)
    )
    # Usamos um dataset separado (mesmos dados, transform de eval) pra
    # validação, pra ela não receber data augmentation — augmentation em
    # validação/teste vazaria ruído artificial na métrica.
    train_full_no_aug = datasets.CIFAR10(
        root=data_dir, train=True, download=False, transform=get_transforms(train=False)
    )
    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=get_transforms(train=False)
    )

    n_val = int(len(train_full) * val_fraction)
    n_train = len(train_full) - n_val

    generator = torch.Generator().manual_seed(seed)
    train_indices, val_indices = random_split(
        range(len(train_full)), [n_train, n_val], generator=generator
    )

    train_dataset = torch.utils.data.Subset(train_full, train_indices.indices)
    val_dataset = torch.utils.data.Subset(train_full_no_aug, val_indices.indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader
