"""
src/train.py

    python -m src.train --epochs 40 --batch-size 128 --lr 0.1

Precisa de internet na primeira execução (baixa o CIFAR-10, ~170MB).
Recomendado rodar com GPU — em CPU, 40 épocas podem levar bastante tempo.
"""

import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn

from src.data import get_dataloaders
from src.engine import evaluate, train_one_epoch
from src.metrics import compute_classification_report, compute_confusion_matrix
from src.model import build_model
from src.utils import save_checkpoint, set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Treina uma ResNet compacta no CIFAR-10")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--patience", type=int, default=8, help="early stopping (épocas sem melhora)")
    p.add_argument("--data-dir", type=str, default="./data")
    p.add_argument("--output-dir", type=str, default="./results")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=2)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    model = build_model(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum,
        weight_decay=args.weight_decay, nesterov=True,
    )
    # OneCycle costuma convergir bem mais rápido que um LR fixo pra CIFAR-10
    # em poucas épocas — sobe o LR no início, desce no final.
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(train_loader)
    )

    best_val_acc = 0.0
    epochs_without_improvement = 0
    history = []

    log_path = output_dir / "training_log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "time_s"])

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, scheduler)
        val_metrics = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - start

        print(
            f"Epoch {epoch}/{args.epochs} — "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.4f} — "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} — "
            f"{elapsed:.1f}s"
        )

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, train_metrics["loss"], train_metrics["accuracy"],
                val_metrics["loss"], val_metrics["accuracy"], round(elapsed, 1),
            ])

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            epochs_without_improvement = 0
            save_checkpoint(model, optimizer, epoch, best_val_acc, str(output_dir / "best_model.pt"))
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            print(f"Early stopping: sem melhora em {args.patience} épocas.")
            break

    # Avaliação final no conjunto de teste, usando o MELHOR checkpoint
    # (não o modelo da última época, que pode ter overfitado depois do pico).
    checkpoint = torch.load(output_dir / "best_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(model, test_loader, criterion, device)

    print(f"\n=== Resultado final no conjunto de TESTE ===")
    print(f"Acurácia: {test_metrics['accuracy']:.4f}")
    print("\n" + compute_classification_report(
        test_metrics["predictions"], test_metrics["labels"],
        class_names=["airplane", "automobile", "bird", "cat", "deer",
                      "dog", "frog", "horse", "ship", "truck"],
    ))

    cm = compute_confusion_matrix(test_metrics["predictions"], test_metrics["labels"])
    import numpy as np
    np.savetxt(output_dir / "confusion_matrix.csv", cm, delimiter=",", fmt="%d")
    print(f"\nMatriz de confusão salva em {output_dir / 'confusion_matrix.csv'}")
    print(f"Melhor checkpoint salvo em {output_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
