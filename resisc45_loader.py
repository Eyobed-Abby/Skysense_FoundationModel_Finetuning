from datasets import load_dataset, concatenate_datasets
from sklearn.model_selection import train_test_split
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset

class HuggingFaceRESISC(Dataset):
    def __init__(self, hf_dataset, transform):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        img = item["image"]
        label = item["label"]
        return self.transform(img), label


def get_resisc45_dataloaders(train_split=0.1, batch_size=64, num_workers=4):
    from torch.utils.data import DataLoader

    # load full dataset
    ds = load_dataset("timm/resisc45")
    full_dataset = concatenate_datasets([ds["train"], ds["test"], ds["validation"]])
    labels = np.array(full_dataset["label"])
    indices = np.arange(len(full_dataset))

    # stratified split by class
    train_idx, test_idx = train_test_split(
        indices, stratify=labels, train_size=train_split, random_state=42
    )
    train_dataset = full_dataset.select(train_idx)
    test_dataset = full_dataset.select(test_idx)

    # define transforms
    img_size = 224
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        transforms.RandomErasing(p=0.5)
    ])
    test_transform = transforms.Compose([
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    # Step 4: wrap and return loaders
    train_data = HuggingFaceRESISC(train_dataset, train_transform)
    test_data = HuggingFaceRESISC(test_dataset, test_transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader


