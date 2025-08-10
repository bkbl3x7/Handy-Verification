import os
import random
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import torch


def find_device(arg: str) -> torch.device:
    if arg != "auto":
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # Apple Silicon
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_min_images_per_subject(df: pd.DataFrame, min_imgs: int = 3) -> pd.DataFrame:
    counts = df.groupby('id')['imageName'].count()
    keep = counts[counts >= min_imgs].index
    return df[df['id'].isin(keep)].copy()


def filter_by_aspect(df: pd.DataFrame, aspect: str) -> pd.DataFrame:
    if aspect == "any":
        return df
    aspect = aspect.lower().strip()
    if aspect == "palmar":
        return df[df['aspectOfHand'].str.contains('palmar', case=False, na=False)]
    if aspect == "dorsal":
        return df[df['aspectOfHand'].str.contains('dorsal', case=False, na=False)]
    return df


def filter_clean(df: pd.DataFrame) -> pd.DataFrame:
    # remove accessories, nail polish, or irregularities if flagged
    if 'accessories' in df and 'nailPolish' in df and 'irregularities' in df:
        return df[(df['accessories'] == 0) & (df['nailPolish'] == 0) & (df['irregularities'] == 0)]
    return df


def stratified_subject_split(subjects: List[int], ratios=(0.7, 0.15, 0.15), seed=42) -> Tuple[List[int], List[int], List[int]]:
    rng = np.random.RandomState(seed)
    subs = np.array(subjects)
    rng.shuffle(subs)
    n = len(subs)
    n_train = int(n * ratios[0])
    n_dev = int(n * ratios[1])
    train = subs[:n_train].tolist()
    dev = subs[n_train:n_train + n_dev].tolist()
    test = subs[n_train + n_dev:].tolist()
    return train, dev, test


def build_image_index(data_dir: str) -> Dict[str, str]:
    # index lowercased filename -> full path
    index: Dict[str, str] = {}
    for root, _, files in os.walk(data_dir):
        for f in files:
            key = f.lower()
            if key not in index:
                index[key] = os.path.join(root, f)
    return index

