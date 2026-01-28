import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from training.model import build_model
from training.trainer import train_model
from training.validate import validate_model

DATASET_PATH = "dataset/synthetic_dataset"
NUM_CLASSES = 8
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

train_tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE,IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5])
])

dataset = datasets.ImageFolder(DATASET_PATH, transform=train_tf)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

model = build_model(NUM_CLASSES)
model = train_model(model, train_loader, DEVICE, EPOCHS)
validate_model(model, val_loader, DEVICE)

torch.save(model.state_dict(), "edge_model.pth")
print("Model saved as edge_model.pth")
