import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from torchvision.transforms import InterpolationMode

img_transform = transforms.Compose(
    [
        transforms.Resize((224, 224), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ]
)

mask_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((224, 224), interpolation=InterpolationMode.NEAREST),
    ]
)

class CustomCOCODataset(Dataset):
    def __init__(self, image_dir, mask_dir, im_transform=None, m_transform=None):
        self.root_dir = os.getcwd()
        self.image_list = sorted(
            [
                os.path.join(self.root_dir, "coco_minitrain2017", img)
                for img in os.listdir(os.path.join(self.root_dir, "coco_minitrain2017"))
            ]
        )
        self.mask_list = sorted(
            [
                os.path.join(self.root_dir, "pseudo_labels", mask)
                for mask in os.listdir(os.path.join(self.root_dir, "pseudo_labels"))
            ]
        )

        self.im_transform = im_transform
        self.m_transform = m_transform

    def __len__(self):
        return len(self.mask_list)

    def __getitem__(self, idx):
        image_path = self.image_list[idx]
        mask_path = self.mask_list[idx]

        image = Image.open(image_path).convert("RGB")
        mask = np.load(mask_path)

        if self.im_transform:
            image = self.im_transform(image)

        if self.m_transform:
            mask = self.m_transform(mask)

        return image, mask.squeeze(0)
