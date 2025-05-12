import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt


# Dataset для правильной обработки меток
class FilteredEMNIST(Dataset):
    def __init__(self, dataset, classes_to_keep):
        self.data = []
        self.targets = []
        for img, label in dataset:
            if label in classes_to_keep:
                self.data.append(img)
                self.targets.append(label - 1)  # конвертируем 1-13 в 0-12

        self.targets = torch.tensor(self.targets)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return len(self.targets)


# модель FFNN
class SimpleFFNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, activation_fn):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.activation_fn = activation_fn
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.activation_fn(self.fc1(x))
        x = self.activation_fn(self.fc2(x))
        x = self.fc3(x)
        return x


def compute_accuracy(model, data_loader):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for images, labels in data_loader:
            outputs = model(images.view(-1, 28 * 28))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


def train_model(train_loader, test_loader, input_size, hidden_size, num_classes, activation_fn, num_epochs=10):
    model = SimpleFFNN(input_size, hidden_size, num_classes, activation_fn)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    train_losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.view(-1, input_size)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        train_acc = compute_accuracy(model, train_loader)
        test_acc = compute_accuracy(model, test_loader)

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"Loss: {train_loss:.4f}, "
              f"Train Acc: {train_acc:.1f}%, "
              f"Test Acc: {test_acc:.1f}%")

    return train_losses, train_accuracies, test_accuracies


def main():
    transform = transforms.Compose([
        transforms.RandomRotation(20),  # поворот ±20°
        transforms.RandomAffine(0, translate=(0.1, 0.1)),  # случайный сдвиг
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # загрузка данных и фильтрация классов A-M (1-13)
    classes_to_keep = list(range(1, 14))  # буквы A-M

    train_data = datasets.EMNIST(root='./data', split='letters', train=True, download=True, transform=transform)
    test_data = datasets.EMNIST(root='./data', split='letters', train=False, download=True, transform=transform)

    train_dataset = FilteredEMNIST(train_data, classes_to_keep)
    test_dataset = FilteredEMNIST(test_data, classes_to_keep)

    # параметры
    input_size = 28 * 28
    hidden_size = 128
    num_classes = 13  # A-M
    num_epochs = 15

    # функции активации
    activation_fns = {
        'ReLU': torch.relu,
        'Sigmoid': torch.sigmoid,
        'Tanh': torch.tanh,
        'LeakyReLU': nn.LeakyReLU(0.1),
        'ELU': nn.ELU()
    }

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # обучение
    results = {}
    for name, activation_fn in activation_fns.items():
        print(f"\n=== {name} ===")
        train_loss, train_acc, test_acc = train_model(
            train_loader, test_loader,
            input_size, hidden_size,
            num_classes, activation_fn, num_epochs
        )
        results[name] = {
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_acc': test_acc
        }

    # визуализация
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    for name, data in results.items():
        plt.plot(data['train_loss'], label=name)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    for name, data in results.items():
        plt.plot(data['test_acc'], label=name)
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()