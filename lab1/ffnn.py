import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# модель FFNN
class SimpleFFNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, activation_fn):
        super(SimpleFFNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.activation_fn = activation_fn

    def forward(self, x):
        x = self.activation_fn(self.fc1(x))
        x = self.activation_fn(self.fc2(x))
        x = self.fc3(x)
        return x


# Функция для вычисления точности
def compute_accuracy(model, data_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


def train_model(train_loader, test_loader, input_size, hidden_size, num_classes, activation_fn, num_epochs=10):
    model = SimpleFFNN(input_size, hidden_size, num_classes, activation_fn)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # потери и точность на обучающих данных
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        train_accuracy = compute_accuracy(model, train_loader)
        train_accuracies.append(train_accuracy)

        # точность на тестовых данных
        test_accuracy = compute_accuracy(model, test_loader)
        test_accuracies.append(test_accuracy)

        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {train_loss:.4f}, "
              f"Train Accuracy: {train_accuracy:.2f}%, "
              f"Test Accuracy: {test_accuracy:.2f}%")

    return train_losses, train_accuracies, test_accuracies


def main():
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.Lambda(lambda x: x.view(-1))  # изображение в вектор
    ])

    train_dataset = datasets.ImageFolder(root='learning_dataset', transform=transform)
    test_dataset = datasets.ImageFolder(root='testing_dataset', transform=transform)

    # параметры
    input_size = 32 * 32
    num_classes = len(train_dataset.classes)
    num_epochs = 10

    # функции активации
    activation_fns = {
        'ReLU': torch.relu,
        'Sigmoid': torch.sigmoid,
        'Tanh': torch.tanh,
        'Leaky ReLU': torch.nn.functional.leaky_relu,
        'ELU': torch.nn.functional.elu
    }

    # размеры скрытого слоя
    hidden_sizes = [64, 128, 256]

    results = {hidden_size: {name: {'train_loss': [], 'train_accuracy': [], 'test_accuracy': []} for name in activation_fns} for hidden_size in hidden_sizes}

    # обучение и тестирование
    for hidden_size in hidden_sizes:
        for name, activation_fn in activation_fns.items():
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

            print(f"Training with {name} and hidden size {hidden_size}...")
            train_losses, train_accuracies, test_accuracies = train_model(train_loader, test_loader, input_size, hidden_size, num_classes, activation_fn, num_epochs)

            results[hidden_size][name]['train_loss'] = train_losses
            results[hidden_size][name]['train_accuracy'] = train_accuracies
            results[hidden_size][name]['test_accuracy'] = test_accuracies

    # графики для каждого hidden_size
    for hidden_size in hidden_sizes:
        plt.figure(figsize=(18, 6))

        # графики потерь
        plt.subplot(1, 3, 1)
        for name, data in results[hidden_size].items():
            plt.plot(data['train_loss'], label=f"{name}")
        plt.xlabel('Epoch')
        plt.ylabel('Training Loss')
        plt.title(f'Training Loss (Hidden Size: {hidden_size})')
        plt.legend()

        # точность на обучающих данных
        plt.subplot(1, 3, 2)
        for name, data in results[hidden_size].items():
            plt.plot(data['train_accuracy'], label=f"{name}")
        plt.xlabel('Epoch')
        plt.ylabel('Training Accuracy')
        plt.title(f'Training Accuracy (Hidden Size: {hidden_size})')
        plt.legend()

        # точность на тестовых данных
        plt.subplot(1, 3, 3)
        for name, data in results[hidden_size].items():
            plt.plot(data['test_accuracy'], label=f"{name}")
        plt.xlabel('Epoch')
        plt.ylabel('Test Accuracy')
        plt.title(f'Test Accuracy (Hidden Size: {hidden_size})')
        plt.legend()

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()