import torch

def validate_model(model, val_loader, device):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            preds = out.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    print("Validation Accuracy:", acc)
    return acc
