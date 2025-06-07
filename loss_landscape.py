def compute_loss_curve(model, loader, criterion, device):
    model.eval()
    losses = []
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        losses.append(loss.item())
    return losses
