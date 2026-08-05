"""
scripts/predict.py

    python -m scripts.predict --checkpoint results/best_model.pt --image caminho/para/imagem.png

Carrega o checkpoint treinado e classifica uma imagem única.
"""

import argparse

import torch
from PIL import Image

from src.data import CIFAR10_MEAN, CIFAR10_STD, CLASS_NAMES, get_transforms
from src.model import build_model


def predict(checkpoint_path: str, image_path: str, device: str = "cpu") -> tuple[str, float]:
    device = torch.device(device)
    model = build_model(num_classes=10).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    image = Image.open(image_path).convert("RGB").resize((32, 32))
    transform = get_transforms(train=False)
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = probs.argmax().item()

    return CLASS_NAMES[pred_idx], probs[pred_idx].item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    label, confidence = predict(args.checkpoint, args.image, args.device)
    print(f"Predição: {label} (confiança: {confidence:.2%})")


if __name__ == "__main__":
    main()
