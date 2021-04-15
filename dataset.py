import os

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import dataset

from PIL import Image


class CelebaDataset(dataset.Dataset):
    def __init__(self, root, transform):
        self.root = root
        self.transform = transform
        self.img_names = os.listdir(root)

    def __len__(self):
        return len(self.img_names)


    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.img_names[idx])
        img = Image.open(img_path)

        return self.transform(img)


def get_loader(img_dir, img_size, batch_size, train='train', num_workers=1):
    
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(178),
            transforms.Resize((img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.CenterCrop(178),
            transforms.Resize((img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ]
    )

    if train == 'train':
        data = CelebaDataset(root=img_dir, transform=train_transform)
        shuffle_data = True
    else:
        data = CelebaDataset(root=img_dir, transform=test_transform)
        shuffle_data = False
    
    data_loader = torch.utils.data.DataLoader(data,
                                            batch_size=batch_size,
                                            shuffle=shuffle_data,
                                            num_workers=num_workers,
                                            drop_last=True)

    return data_loader

