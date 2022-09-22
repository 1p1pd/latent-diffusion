import os
import numpy as np
import PIL
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class TexBase(Dataset):
    def __init__(self,
                 img_path='',
                 size=None,
                 n_img=0,
                 ):
        self.img_path = img_path
        self.size = size
        self.n_img = n_img

    def __len__(self):
        return self.n_img

    def __getitem__(self, i):
        data_aug = transforms.Compose(
            [transforms.RandomCrop(self.size),
             transforms.RandomHorizontalFlip(),
             transforms.RandomVerticalFlip(),
             ])
        image = Image.open(self.img_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = data_aug(image)
        image = np.array(image).astype(np.float32) / 255.

        mask = np.zeros((self.size, self.size, 3), dtype=np.float32)
        if i % 4 == 0:
            mask[:self.size // 2, :, :] = 1.
        elif i % 4 == 1:
            mask[self.size // 2:, :, :] = 1.
        elif i % 4 == 2:
            mask[:, :self.size // 2, :] = 1.
        elif i % 4 == 3:
            mask[:, self.size // 2:, :] = 1.

        masked_image = (1. - mask) * image

        batch = {"image": image, "mask": mask, "masked_image": masked_image}
        for k in batch:
            batch[k] = batch[k] * 2.0 - 1.0
        return batch


class TexTrain(TexBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class TexValidation(TexBase):
    def __init__(self, flip_p=0., **kwargs):
        super().__init__(**kwargs)


class TexQuadBase(Dataset):
    def __init__(self,
                 img_path='',
                 size=None,
                 n_img=0,
                 ):
        self.img_path = img_path
        self.size = size
        self.n_img = n_img

    def __len__(self):
        return self.n_img

    def __getitem__(self, i):
        data_aug = transforms.Compose(
            [transforms.RandomCrop(self.size),
             transforms.RandomHorizontalFlip(),
             transforms.RandomVerticalFlip(),
             ])
        image = Image.open(self.img_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = data_aug(image)
        image = np.array(image).astype(np.float32) / 255.

        mask = np.zeros((self.size, self.size, 3), dtype=np.float32)
        n_rand = torch.randint(1, 5, (1,)).item()
        rand_block = torch.randperm(4)[:n_rand]
        for i in rand_block:
            if i == 0:
                mask[:self.size//2, :self.size//2, :] = 1.
            elif i == 1:
                mask[self.size//2:, :self.size//2, :] = 1.
            elif i == 2:
                mask[:self.size//2, self.size//2:, :] = 1.
            elif i == 3:
                mask[self.size//2:, self.size//2:, :] = 1.

        masked_image = (1. - mask) * image

        batch = {"image": image, "mask": mask, "masked_image": masked_image}
        for k in batch:
            batch[k] = batch[k] * 2.0 - 1.0
        return batch


class TexQuadTrain(TexQuadBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class TexQuadValidation(TexQuadBase):
    def __init__(self, flip_p=0., **kwargs):
        super().__init__(**kwargs)