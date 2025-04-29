import os

import kagglehub
from torchvision.datasets import ImageFolder

# Download latest version
path = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

transform = transforms.Compose(
    [
        transforms.Resize((512, 512)),  # Resize images to 224x224
        transforms.ToTensor(),  # Convert to tensor (scales pixels to [0,1])
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),  # ImageNet normalization
    ]
)

import os

image_train = os.path.join(path, "Training")
image_test = os.path.join(path, "Testing")
train_dataset = ImageFolder(image_train, transform=transform)
test_dataset = ImageFolder(image_test, transform=transform)
