import os

import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from src.datasets.crack_dataset import CrackDataset


def get_transforms():
    train_transform = A.Compose([
        A.Resize(256, 256),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    return train_transform, val_transform


def dice_loss(preds, targets, smooth=1.0):
    preds = torch.sigmoid(preds)
    preds = preds.contiguous()
    targets = targets.contiguous()

    intersection = (preds * targets).sum(dim=(2, 3))
    union = preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))

    loss = 1 - ((2.0 * intersection + smooth) / (union + smooth))
    return loss.mean()


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = dice_loss(logits, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate_one_epoch(model, loader, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            loss = dice_loss(logits, masks)
            total_loss += loss.item()

    return total_loss / len(loader)


def save_loss_plot(train_losses, val_losses, save_path):
    epochs = list(range(1, len(train_losses) + 1))

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_loss_plot_every_10(train_losses, val_losses, save_path):
    epochs = list(range(1, len(train_losses) + 1))

    sampled_epochs = []
    sampled_train_losses = []
    sampled_val_losses = []

    for i, epoch in enumerate(epochs):
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs[-1]:
            sampled_epochs.append(epoch)
            sampled_train_losses.append(train_losses[i])
            sampled_val_losses.append(val_losses[i])

    plt.figure(figsize=(10, 6))
    plt.plot(sampled_epochs, sampled_train_losses, marker="o", label="Train Loss")
    plt.plot(sampled_epochs, sampled_val_losses, marker="o", label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss (Every 10 Epochs)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_loss_history(train_losses, val_losses, save_path):
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,val_loss\n")
        for epoch, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses), start=1):
            f.write(f"{epoch},{train_loss:.6f},{val_loss:.6f}\n")


def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    images_dir = os.path.join(base_dir, "data", "images")
    masks_dir = os.path.join(base_dir, "data", "masks")
    models_dir = os.path.join(base_dir, "src", "models")

    batch_size = 4
    num_epochs = 50
    learning_rate = 1e-3

    os.makedirs(models_dir, exist_ok=True)

    best_model_path = os.path.join(models_dir, "best_model.pth")
    final_model_path = os.path.join(models_dir, "final_model.pth")
    loss_csv_path = os.path.join(models_dir, "loss_history.csv")
    full_plot_path = os.path.join(models_dir, "loss_curve_full.png")
    sampled_plot_path = os.path.join(models_dir, "loss_curve_every_10_epochs.png")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_transform, val_transform = get_transforms()

    full_dataset = CrackDataset(images_dir=images_dir, masks_dir=masks_dir, transform=None)

    indices = list(range(len(full_dataset)))
    train_indices, val_indices = train_test_split(
        indices, test_size=0.2, random_state=42
    )

    train_dataset = CrackDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        transform=train_transform,
    )
    val_dataset = CrackDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        transform=val_transform,
    )

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float("inf")
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = validate_one_epoch(model, val_loader, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}] - "
                f"Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(
                f"Saved best model at epoch {epoch + 1} "
                f"with validation loss {val_loss:.4f}"
            )

    torch.save(model.state_dict(), final_model_path)
    print(f"Saved final model to: {final_model_path}")

    save_loss_history(train_losses, val_losses, loss_csv_path)
    save_loss_plot(train_losses, val_losses, full_plot_path)
    save_loss_plot_every_10(train_losses, val_losses, sampled_plot_path)

    print(f"Saved loss history to: {loss_csv_path}")
    print(f"Saved full loss plot to: {full_plot_path}")
    print(f"Saved sampled loss plot to: {sampled_plot_path}")
    print("Training complete.")


if __name__ == "__main__":
    main()