import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import wandb
import os
import argparse

from resisc45_loader import get_resisc45_dataloaders
from skysense_lora_classifier_qkv import build_lora_classifier_qkv


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for x, y in tqdm(dataloader, desc="Train", leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)

    return running_loss / total, correct / total


def evaluate(model, dataloader, criterion, device):
    model.eval()
    loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Val", leave=False):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss += criterion(logits, y).item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)

    return loss / total, correct / total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=6.25e-5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--train_split', type=float, default=0.1)
    parser.add_argument('--save_path', type=str, default="checkpoints/best_model.pth")
    parser.add_argument('--wandb_project', type=str, default="LoRA_expt")
    args = parser.parse_args()

    wandb.init(project=args.wandb_project, config=vars(args))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_lora_classifier_qkv(args.ckpt).to(device)

    train_loader, val_loader = get_resisc45_dataloaders(train_split=args.train_split, batch_size=args.batch_size)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        scheduler.step()

        wandb.log({
            "epoch": epoch+1,
            "train/loss": train_loss,
            "train/acc": train_acc,
            "val/loss": val_loss,
            "val/acc": val_acc,
            "lr": optimizer.param_groups[0]['lr']
        })

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), args.save_path)
            print(f"Saved new best model @ epoch {epoch+1} with val_acc={val_acc:.4f}")
