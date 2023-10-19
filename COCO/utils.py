import os
import numpy as np
import torch
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as F

img_transform = transforms.Compose(
    [
        transforms.Resize((256, 256), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ]
)

mask_transform = transforms.Compose(
    [
        transforms.Resize((256, 256), interpolation=InterpolationMode.NEAREST),
    ]
)


class CustomCOCODataset(Dataset):
    def __init__(self, image_dir, mask_dir, im_transform, m_transform):
        self.root_dir = os.getcwd()
        self.image_list = sorted(
            [
                os.path.join(self.root_dir, image_dir, img)
                for img in os.listdir(os.path.join(self.root_dir, image_dir))
            ]
        )
        self.mask_list = sorted(
            [
                os.path.join(self.root_dir, mask_dir, mask)
                for mask in os.listdir(os.path.join(self.root_dir, mask_dir))
            ]
        )

        self.im_transform = im_transform
        self.m_transform = m_transform

    def transform(self, image, mask):
        if random.random() > 0.5:
            i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(256, 256))
            image = F.crop(image, i, j, h, w)
            mask = F.crop(mask, i, j, h, w)

        if random.random() > 0.5:
            image = F.hflip(image)
            mask = F.hflip(mask)

        if random.random() > 0.5:
            image = F.vflip(image)
            mask = F.vflip(mask)

        return image, mask

    def __len__(self):
        return len(self.mask_list)

    def __getitem__(self, idx):
        image_path = self.image_list[idx]
        mask_path = self.mask_list[idx]

        image = Image.open(image_path).convert("RGB")
        mask = np.load(mask_path).astype("int64")
        mask = torch.from_numpy(mask).unsqueeze(0)

        image = self.im_transform(image)
        mask = self.m_transform(mask)
        image, mask = self.transform(image, mask)

        return image, mask.squeeze(0)
