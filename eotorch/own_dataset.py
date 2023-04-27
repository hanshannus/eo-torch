from typing import Any
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import numpy as np
from torchvision import transforms as T
import torch


def to_tuple(n):
    if isinstance(n, (int, float)):
        return (n, n)
    return n


class ImageCropDataset(Dataset):

    def __init__(self,
                 image_base_dir,
                 size=256,
                 transform=None
                 ):
        self.image_base_dir = Path(image_base_dir)
        self.images = list(self.image_base_dir.glob("Images/*.jpg"))
        crops = {}
        self.size = to_tuple(size)
        self.to_tensor = T.Compose([
            T.ToTensor()
        ])
        self.transform = transform
        for image in self.images:
            if not (self.image_base_dir / "Masks" / image.name).with_suffix(".png").exists():
                continue
            width, height = Image.open(image).size
            max_h = height - self.size[0]
            max_w = width - self.size[1]
            steps_h = int(np.ceil(max_h / self.size[0]))
            steps_w = int(np.ceil(max_w / self.size[1]))
            crops[image.name] = []
            for top in np.linspace(0, max_h, steps_h, dtype=int):
                bottom = top + self.size[0]
                for left in np.linspace(0, max_w, steps_w, dtype=int):
                    right = left + self.size[1]
                    crops[image.name].append((top, left, bottom, right))
        self.data = []
        for image in crops:
            for crop in crops[image]:
                self.data.append(
                    (self.image_base_dir / "Images" / image,
                     (self.image_base_dir / "Masks" / image).with_suffix(".png"),
                     crop)
                )

        self._cache = {}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int) -> Any:
        image, mask, (top, left, bottom, right) = self.data[index]
        image_name = image
        if image_name not in self._cache:
            image = self.to_tensor(Image.open(image).convert("RGB"))
            mask = torch.tensor(np.array(Image.open(mask)))
            if image[0].shape != mask.shape:
                mask = mask.rot90(1, [0, 1])
            self._cache[image_name] = image, mask
        else:
            image, mask = self._cache[image_name]
        img, mask = (image[..., top:bottom, left:right],
                     mask[..., top:bottom, left:right])
        if self.transform is not None:
            img = self.transform(img)
        return {"image": img, "mask": mask}


def get_random_bounding_box(size, height, width):
    y = np.random.randint(0, height - size[0])
    x = np.random.randint(0, width - size[1])
    return y, x, y + size[0], x + size[1]


class RandomImageCropDataset(Dataset):

    def __init__(self,
                 image_base_dir,
                 crops_per_image=100,
                 size=256,
                 transform=None
                 ):
        self.image_base_dir = Path(image_base_dir)
        images = list(self.image_base_dir.glob("Images/*.jpg"))
        self.images = []
        for image in images:
            if not (self.image_base_dir / "Masks" / image.name).with_suffix(".png").exists():
                continue
            self.images.append(image)
        self.size = to_tuple(size)
        self.crops_per_image = crops_per_image
        self.to_tensor = T.Compose([
            T.ToTensor()
        ])
        self._cache = {}
        self.transform = transform

    def __len__(self):
        return len(self.images) * self.crops_per_image

    def __getitem__(self, index: int) -> Any:
        index = index // self.crops_per_image
        image = self.images[index]
        image_name = image.name
        if image_name not in self._cache:
            mask = (self.image_base_dir / "Masks" /
                    image_name).with_suffix(".png")
            image = self.to_tensor(Image.open(image).convert("RGB"))
            mask = torch.tensor(np.array(Image.open(mask)))
            if image[0].shape != mask.shape:
                mask = mask.rot90(1, [0, 1])
            self._cache[image_name] = image, mask
        else:
            image, mask = self._cache[image_name]

        top, left, bottom, right = get_random_bounding_box(
            self.size, *image.shape[-2:])
        img, mask = (image[..., top:bottom, left:right],
                     mask[..., top:bottom, left:right])
        if self.transform is not None:
            img = self.transform(img)
        return {"image": img, "mask": mask}
