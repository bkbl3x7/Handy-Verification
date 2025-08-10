from typing import Dict

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import Sampler
import numpy as np


class HandDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, img_index: Dict[str, str], transform=None):
        self.rows = rows.reset_index(drop=True)
        self.img_index = img_index
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows.iloc[idx]
        fname = str(r['imageName']).lower()
        path = self.img_index.get(fname, None)
        if path is None:
            raise FileNotFoundError(f"Image not found in data_dir for {r['imageName']}")
        with Image.open(path) as im:
            im = im.convert('RGB')
        if self.transform:
            im = self.transform(im)
        label = int(r['id'])
        return im, label


def make_transforms(img_size: int, aug: bool):
    if aug:
        train_tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.RandomRotation(8, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.ToTensor()
        ])
    else:
        train_tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])
    return train_tf, eval_tf




class BalancedPKSampler(Sampler):
    """
    Yields batches of indices shaped as P identities × K images each.
    Greatly improves triplet mining.
    """
    def __init__(self, labels, P=16, K=4, seed=42):
        self.labels = np.array(labels, dtype=int)
        self.P, self.K = P, K
        self.rng = np.random.RandomState(seed)
        self.by_label = {}
        for idx, y in enumerate(self.labels.tolist()):
            self.by_label.setdefault(y, []).append(idx)
        self.ids = list(self.by_label.keys())
        self.batch_size = P*K
        self.num_batches = max(1, len(self.labels) // self.batch_size)

    def __len__(self): return self.num_batches

    def __iter__(self):
        for _ in range(self.num_batches):
            P = min(self.P, len(self.ids))
            chosen = self.rng.choice(self.ids, size=P, replace=False)
            batch = []
            for y in chosen:
                pool = self.by_label[y]
                if len(pool) >= self.K:
                    pick = self.rng.choice(pool, size=self.K, replace=False)
                else:
                    pick = self.rng.choice(pool, size=self.K, replace=True)
                batch.extend(pick.tolist())
            yield batch


