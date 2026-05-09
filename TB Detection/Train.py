"""
Train.py — EfficientNet-B3 TB Detector Training Script
Place images in:
  data/normal/         → normal chest X-rays
  data/tuberculosis/   → TB positive X-rays
Then run: python Train.py
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import os
import time


def train_model(data_dir='data', epochs=50, batch_size=16, lr=0.0001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cpu':
        print("Note: Training on CPU will be slow. Consider Google Colab for GPU training.")

    train_transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    full_dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)
    print(f"Classes: {full_dataset.classes}")
    print(f"Class→idx: {full_dataset.class_to_idx}")

    train_size = int(0.8 * len(full_dataset))
    val_size   = len(full_dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    val_ds.dataset.transform = val_transform

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # ── Model ──────────────────────────────────────────────────────────────────
    print("Loading EfficientNet-B3 ...")
    try:
        model = models.efficientnet_b3(weights='IMAGENET1K_V1')
    except Exception:
        model = models.efficientnet_b3(weights=None)

    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 1),
        nn.Sigmoid(),
    )
    model = model.to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    os.makedirs('models', exist_ok=True)
    best_val_loss = float('inf')

    print("\nStarting training …")
    for epoch in range(epochs):
        t0 = time.time()

        # Training
        model.train()
        train_loss = correct = total = 0
        for bi, (imgs, lbls) in enumerate(train_loader):
            imgs = imgs.to(device)
            lbls = lbls.float().unsqueeze(1).to(device)
            optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, lbls)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            correct    += ((out > 0.5).float() == lbls).sum().item()
            total      += lbls.size(0)
            if (bi + 1) % 5 == 0:
                print(f"  Batch {bi+1}/{len(train_loader)}  loss={loss.item():.4f}")

        # Validation
        model.eval()
        val_loss = vc = vt = 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs = imgs.to(device)
                lbls = lbls.float().unsqueeze(1).to(device)
                out  = model(imgs)
                val_loss += criterion(out, lbls).item()
                vc += ((out > 0.5).float() == lbls).sum().item()
                vt += lbls.size(0)

        avg_tl = train_loss / len(train_loader)
        avg_vl = val_loss   / len(val_loader)
        print(f"\nEpoch [{epoch+1}/{epochs}]  {time.time()-t0:.1f}s")
        print(f"  Train  loss={avg_tl:.4f}  acc={100*correct/total:.2f}%")
        print(f"  Val    loss={avg_vl:.4f}  acc={100*vc/vt:.2f}%")

        scheduler.step(avg_vl)

        if avg_vl < best_val_loss:
            best_val_loss = avg_vl
            torch.save(model.state_dict(), 'models/tb_model.pth')
            print("  ✓ Best model saved → models/tb_model.pth")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    train_model("data/", epochs=50, batch_size=16, lr=0.0001)
