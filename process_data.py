import os

import kagglehub
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder

dataset_path = (
    "/root/.cache/kagglehub/datasets/paultimothymooney/chest-xray-pneumonia/versions/2"
)
# Download latest version
path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
train_path = os.path.join(
    dataset_path, "chest_xray/train"
)  # Adjust if folder name differs
test_path = os.path.join(dataset_path, "chest_xray/test")

print("Train folder contents:", os.listdir(train_path))
print("Test folder contents:", os.listdir(test_path))
transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
train_path = "/root/.cache/kagglehub/datasets/paultimothymooney/chest-xray-pneumonia/versions/2/chest_xray/train"
test_path = "/root/.cache/kagglehub/datasets/paultimothymooney/chest-xray-pneumonia/versions/2/chest_xray/test"
val_path = "/root/.cache/kagglehub/datasets/paultimothymooney/chest-xray-pneumonia/versions/2/chest_xray/val"

train_dataset = ImageFolder(root=train_path, transform=transforms)
test_dataset = ImageFolder(root=test_path, transform=transforms)

val_dataset = ImageFolder(root=val_path, transform=transforms) if val_path else None
