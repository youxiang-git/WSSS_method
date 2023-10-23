import os
import numpy as np
import torch
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as F


class CustomCRFDataset(Dataset):
    def __init__(self, voc_root, set="train", transform=False, weak=False):
        self.curr_dir = os.getcwd()
        self.weak = weak
        self.set = set
        self.transform = transform
        self.voc_root = os.path.join(self.curr_dir, voc_root, "VOCdevkit", "VOC2012")

        if not os.path.isdir(voc_root):
            raise RuntimeError("VOC Directory is wrong")

        splits_dir = os.path.join(self.voc_root, "ImageSets", "Segmentation")

        if self.set == "train" or self.set == "val":
            splits_f = os.path.join(splits_dir, "{}.txt".format(self.set))
        else:
            raise RuntimeError("Set must be train or val")
        
        # print("I'm the {} set".format(self.set))

        with open(os.path.join(splits_f)) as f:
            file_names = [x.strip() for x in f.readlines()]

        if self.weak:
            seg_dir = os.path.join(self.curr_dir, "voc_pseudomasks_Gdino_SAM")
        else:
            seg_dir = os.path.join(self.voc_root, "SegmentationClass")

        img_dir = os.path.join(self.voc_root, "JPEGImages")

        self.images = [os.path.join(img_dir, x + ".jpg") for x in file_names]
        self.masks = [os.path.join(seg_dir, x + ".png") for x in file_names]

        # print("hello ", self.images[0])
        # print("hello!! ", self.masks[0])

        assert len(self.images) == len(self.masks)

    def transformations(self, image, mask):
        # if random.random() > 0.8 and image.size[0] > 256 and image.size[1] > 256:
        #     i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(256, 256))
        #     image = F.crop(image, i, j, h, w)
        #     mask = F.crop(mask, i, j, h, w)

        if random.random() > 0.5:
            image = F.hflip(image)
            mask = F.hflip(mask)

        if random.random() > 0.5:
            image = F.vflip(image)
            mask = F.vflip(mask)

        return image, mask

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        img_pure = F.pil_to_tensor(Image.open(self.images[idx]).convert("RGB"))
        mask = Image.open(self.masks[idx])

        if self.transform:
            img, mask = self.transformations(img, mask)

        img = F.resize(img, [256, 256], InterpolationMode.BILINEAR)
        mask = F.resize(mask, [256, 256], InterpolationMode.NEAREST)

        img = F.to_tensor(img)
        img = F.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        mask = F.pil_to_tensor(mask).squeeze()
        mask = mask.type(torch.LongTensor)

        return img, mask, img_pure
