"""Train a small neural network to classify MNIST handwritten digits."""

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DATA_DIR = Path(__file__).parent / "data"
BATCH_SIZE = 64
EPOCHS = 3
LEARNING_RATE = 1e-3


def get_device() -> torch.device:
    """Use the GPU if there is one (Apple Silicon = 'mps'), otherwise the CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_loaders() -> tuple[DataLoader, DataLoader]:
    """Download MNIST and wrap it in DataLoaders that serve batches."""
    transform = transforms.ToTensor()
    train_set = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1000)
    return train_loader, test_loader


class DigitClassifier(nn.Module):
    """A small fully connected network: 784 pixels -> 128 hidden -> 10 digits."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Go through the training data once, updating the model. Returns avg loss."""
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        predictions = model(images)
        loss = loss_fn(predictions, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Return the model's accuracy (0 to 1) on data it has never trained on."""
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            predicted_digits = model(images).argmax(dim=1)
            correct += (predicted_digits == labels).sum().item()
    return correct / len(loader.dataset)


def main() -> None:
    device = get_device()
    print(f"Using device: {device}")

    train_loader, test_loader = get_loaders()
    model = DigitClassifier().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(1, EPOCHS + 1):
        avg_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        accuracy = evaluate(model, test_loader, device)
        print(f"Epoch {epoch}: loss = {avg_loss:.4f}, test accuracy = {accuracy:.2%}")


if __name__ == "__main__":
    main()
