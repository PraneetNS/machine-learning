import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import os

class CustomImageDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.data.iloc[idx, 0])
        image = Image.open(img_path).convert("RGB")
        label = self.data.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        return image, label


# Usage
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

dataset = CustomImageDataset("data.csv", "images/", transform)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for images, labels in loader:
    print(images.shape, labels.shape)
    break