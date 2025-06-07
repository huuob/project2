from vgg import VGG_A, VGG_BN
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def run_bn_experiment():
    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    loader = DataLoader(trainset, batch_size=128, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = [VGG_A().to(device), VGG_BN().to(device)]
    names = ['VGG-A', 'VGG-A-BN']
    results = []

    for model in models:
        losses = []
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        for epoch in range(5):
            epoch_losses = []
            for inputs, targets in loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())
            losses.append(epoch_losses)
        results.append(losses)

    from visualize_loss import visualize_loss_curves
    visualize_loss_curves(results, names)


if __name__ == "__main__":
    run_bn_experiment()
