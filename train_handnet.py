
#!/usr/bin/env python3
"""
train_handnet.py

Minimal, reproducible training pipeline for hand verification.
- Works on Apple Silicon (MPS) or CPU (and CUDA if available).
- Supports a small custom CNN ("handnet") or ResNet18 ("resnet18").
- Loss: cross-entropy (ID classification) or triplet margin.
- Evaluation: EER/AUC with 2-image enrollment and cosine similarity.
- Saves: best weights (by dev EER), scores.csv, metrics.json, roc.png

Usage examples
--------------
# Train a ResNet18 baseline with CE on palmar hands only
python train_handnet.py --data_dir ~/hands/images --csv_path ~/hands/HandInfo.csv \
  --backbone resnet18 --loss ce --aspect palmar --epochs 15 --batch_size 64 --img_size 224 --device auto

# Train custom CNN with triplet loss on dorsal hands
python train_handnet.py --data_dir ~/hands/images --csv_path ~/hands/HandInfo.csv \
  --backbone handnet --loss triplet --aspect dorsal --epochs 25 --batch_size 96 --img_size 224 --device auto

# Evaluate only (given weights file)
python train_handnet.py --data_dir ~/hands/images --csv_path ~/hands/HandInfo.csv \
  --backbone resnet18 --eval_only --weights runs/best.pt --aspect palmar --img_size 224 --device auto
"""
import os, random, argparse, math, json, csv, time
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision import transforms

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# ---------------------------
# Utilities
# ---------------------------
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
    n_dev   = int(n * ratios[1])
    train = subs[:n_train].tolist()
    dev   = subs[n_train:n_train+n_dev].tolist()
    test  = subs[n_train+n_dev:].tolist()
    return train, dev, test

def build_image_index(data_dir: str) -> Dict[str, str]:
    # index lowercased filename -> full path
    index = {}
    for root, _, files in os.walk(data_dir):
        for f in files:
            key = f.lower()
            if key not in index:
                index[key] = os.path.join(root, f)
    return index

# ---------------------------
# Dataset
# ---------------------------
class HandDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, img_index: Dict[str,str], transform=None):
        self.rows = rows.reset_index(drop=True)
        self.img_index = img_index
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows.iloc[idx]
        fname = str(r['imageName']).lower()
        path  = self.img_index.get(fname, None)
        if path is None:
            raise FileNotFoundError(f"Image not found in data_dir for {r['imageName']}")
        with Image.open(path) as im:
            im = im.convert('RGB')
        if self.transform:
            im = self.transform(im)
        label = int(r['id'])
        return im, label

# ---------------------------
# Models
# ---------------------------
class HandNetSmall(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)
        self.pool  = nn.AdaptiveAvgPool2d((1,1))
        self.fc    = nn.Linear(128, embed_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)         # 112x112
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)         # 56x56
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)         # 28x28
        x = self.pool(x).view(x.size(0), -1)  # (B,128)
        x = F.normalize(self.fc(x))    # L2-normalized embedding (B,embed_dim)
        return x

def make_backbone(name: str, embed_dim=128):
    if name == "handnet":
        return HandNetSmall(embed_dim)
    if name == "resnet18":
        m = torchvision.models.resnet18(weights="IMAGENET1K_V1")
        # replace fc with embed head
        in_f = m.fc.in_features
        m.fc = nn.Sequential(
            nn.Linear(in_f, embed_dim),
            nn.BatchNorm1d(embed_dim)
        )
        # normalize in forward via wrapper
        class ResnetWrap(nn.Module):
            def __init__(self, m):
                super().__init__(); self.m = m
            def forward(self, x):
                z = self.m(x)
                return F.normalize(z)
        return ResnetWrap(m)
    raise ValueError("Unknown backbone")

# CE head for ID classification (emb -> logits).
class LinearClassifier(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)
    def forward(self, x):
        return self.fc(x)

# ---------------------------
# Triplet mining (batch-hard lite)
# ---------------------------
def batch_hard_triplet_loss(embeddings: torch.Tensor, labels: torch.Tensor, margin=0.2):
    # embeddings: (B,D), labels: (B,)
    B = embeddings.size(0)
    # Compute pairwise distances
    dist = torch.cdist(embeddings, embeddings, p=2)
    loss = 0.0
    count = 0
    for i in range(B):
        anchor_label = labels[i]
        pos_mask = (labels == anchor_label)
        neg_mask = ~pos_mask
        pos_mask[i] = False
        if pos_mask.any() and neg_mask.any():
            hardest_pos = dist[i][pos_mask].max()
            hardest_neg = dist[i][neg_mask].min()
            l = F.relu(hardest_pos - hardest_neg + margin)
            loss += l
            count += 1
    if count == 0:
        return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
    return loss / count

# ---------------------------
# Enrollment & EER evaluation
# ---------------------------
def embed_loader(model, loader, device):
    model.eval()
    all_z, all_y = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            z = model(x)
            all_z.append(z.cpu())
            all_y.append(y)
    return torch.cat(all_z, 0), torch.cat(all_y, 0)

def mean_enroll(embeds: torch.Tensor, labels: torch.Tensor, per_subject=2):
    # For each subject: first `per_subject` as enroll, rest as probes
    enroll = {}
    probes = []
    # Group indices by label preserving order
    by_label: Dict[int, List[int]] = {}
    for i, y in enumerate(labels.tolist()):
        by_label.setdefault(y, []).append(i)
    for y, idxs in by_label.items():
        if len(idxs) < per_subject + 1:
            continue
        e_idx = idxs[:per_subject]
        p_idx = idxs[per_subject:]
        e = embeds[e_idx].mean(0, keepdim=True)  # (1,D)
        for j in p_idx:
            probes.append((e, embeds[j], 1))  # genuine
    # Impostors: sample against other subjects' enroll means
    enroll_means = []
    for y, idxs in by_label.items():
        if len(idxs) >= per_subject + 1:
            e_idx = idxs[:per_subject]
            enroll_means.append(embeds[e_idx].mean(0, keepdim=True))
    if len(enroll_means) < 2:
        return np.array([]), np.array([])  # not enough
    enroll_means = torch.cat(enroll_means, 0)  # (S,D)

    # Build genuine scores collected above, and impostors by comparing each probe to a random *other* enroll mean
    genu_s, imp_s = [], []
    for e, p, _ in probes:
        # cosine similarity
        gs = F.cosine_similarity(e, p.unsqueeze(0)).item()
        genu_s.append(gs)
        # impostor: compare p to a *different* subject's enroll mean
        # take the hardest impostor among a small random subset for stability
        with torch.no_grad():
            sims = F.cosine_similarity(enroll_means, p.unsqueeze(0).expand_as(enroll_means))
            imp = torch.max(sims[:-1]) if sims.size(0) > 1 else sims[0]  # approximate "hard" impostor
            imp_s.append(float(imp))

    y_true = np.array([1]*len(genu_s) + [0]*len(imp_s))
    y_score = np.array(genu_s + imp_s)
    return y_true, y_score

def compute_eer(y_true: np.ndarray, y_score: np.ndarray):
    if len(y_true) == 0:
        return dict(eer=np.nan, thr=np.nan, auc=np.nan)
    fpr, tpr, thr = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    idx = int(np.nanargmin(np.abs(fnr - fpr)))
    eer = float((fnr[idx] + fpr[idx]) / 2)
    auc = float(roc_auc_score(y_true, y_score))
    return dict(eer=eer, thr=float(thr[idx]), auc=auc, fpr=fpr, tpr=tpr, thr_all=thr)

def plot_roc(fpr, tpr, auc, eer=None, out_path="roc.png"):
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC (AUC={auc:.3f})')
    plt.plot([0,1], [0,1], linestyle='--', label='Chance')
    if eer is not None:
        # find point nearest to EER on the curve for plotting
        diffs = np.abs((1 - tpr) - fpr)
        j = int(np.nanargmin(diffs))
        plt.scatter([fpr[j]], [tpr[j]], marker='o', label=f'EER≈{eer*100:.1f}%')
    plt.xlabel('False Positive Rate (FMR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()

# ---------------------------
# Training
# ---------------------------
def train_one_epoch(model, clf, loader, optimizer, device, loss_type="ce"):
    model.train()
    if clf is not None:
        clf.train()
    losses = []
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()
        z = model(xb)            # (B, D), normalized
        if loss_type == "ce":
            logits = clf(z)
            loss = F.cross_entropy(logits, yb)
        else:
            loss = batch_hard_triplet_loss(z, yb, margin=0.2)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else 0.0

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

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--csv_path", required=True)
    p.add_argument("--aspect", default="palmar", choices=["palmar","dorsal","any"])
    p.add_argument("--clean_only", action="store_true", help="Drop accessories/nailPolish/irregularities if present")
    p.add_argument("--min_images", type=int, default=3)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--backbone", default="resnet18", choices=["resnet18","handnet"])
    p.add_argument("--loss", default="ce", choices=["ce","triplet"])
    p.add_argument("--device", default="auto", help="auto|cpu|cuda|mps")
    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--weights", default="", help="Path to weights for eval_only")
    p.add_argument("--out_dir", default="runs")
    args = p.parse_args()

    set_seed(args.seed)
    device = find_device(args.device)
    print(f"Using device: {device}")

    # Load metadata
    df = pd.read_csv(args.csv_path)
    df = filter_by_aspect(df, args.aspect)
    if args.clean_only:
        df = filter_clean(df)
    df = ensure_min_images_per_subject(df, args.min_images)

    # Index images
    img_index = build_image_index(args.data_dir)
    print(f"Indexed {len(img_index)} files under {args.data_dir}")
    print(f"Rows after filtering: {len(df)}; Subjects: {df['id'].nunique()}")

    # Subject split
    subjects = sorted(df['id'].unique().tolist())
    train_sub, dev_sub, test_sub = stratified_subject_split(subjects, ratios=(0.7,0.15,0.15), seed=args.seed)
    df_train = df[df['id'].isin(train_sub)].copy()
    df_dev   = df[df['id'].isin(dev_sub)].copy()
    df_test  = df[df['id'].isin(test_sub)].copy()
    print(f"Split → train/dev/test subjects: {len(train_sub)}/{len(dev_sub)}/{len(test_sub)}")

    # Transforms
    train_tf, eval_tf = make_transforms(args.img_size, aug=True)

    # Label remap for classifier
    id_to_idx = {sid:i for i, sid in enumerate(sorted(df_train['id'].unique()))}
    def remap(df):
        return df.assign(lbl=df['id'].map(id_to_idx).fillna(-1).astype(int))

    df_train = remap(df_train)
    df_dev   = remap(df_dev)
    df_test  = remap(df_test)

    # Datasets
    train_ds = HandDataset(df_train[['imageName','id']], img_index, transform=train_tf)
    dev_ds   = HandDataset(df_dev[['imageName','id']], img_index, transform=eval_tf)
    test_ds  = HandDataset(df_test[['imageName','id']], img_index, transform=eval_tf)

    # Loaders (for CE we benefit from class-balanced sampling, but keep simple here)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    dev_loader   = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Model
    embed_dim = 128
    model = make_backbone(args.backbone, embed_dim=embed_dim).to(device)

    clf = None
    if args.loss == "ce":
        clf = LinearClassifier(embed_dim, num_classes=len(id_to_idx)).to(device)
        params = list(model.parameters()) + list(clf.parameters())
    else:
        params = list(model.parameters())

    # Optimizer
    opt = torch.optim.Adam(params, lr=1e-3, weight_decay=1e-4)

    # Out dir
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    run_tag = f"{args.backbone}_{args.loss}_{args.aspect}_{time.strftime('%Y%m%d-%H%M%S')}"
    out_dir = Path(args.out_dir) / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # Eval-only
    if args.eval_only:
        assert args.weights, "--weights required for --eval_only"
        state = torch.load(args.weights, map_location=device)
        model.load_state_dict(state['model'])
        if 'clf' in state and clf is not None and state['clf'] is not None:
            clf.load_state_dict(state['clf'])
        evaluate_and_save(model, dev_loader, test_loader, out_dir)
        return

    # Train loop
    best_eer = 1.0
    best_state = None
    for epoch in range(1, args.epochs+1):
        loss_val = train_one_epoch(model, clf, train_loader, opt, device, loss_type=args.loss)
        # quick EER on dev
        dev_eer = evaluate(model, dev_loader)
        print(f"Epoch {epoch:02d} | loss={loss_val:.4f} | dev EER={dev_eer*100:.2f}%")
        if dev_eer < best_eer:
            best_eer = dev_eer
            best_state = {
                'model': model.state_dict(),
                'clf': clf.state_dict() if clf is not None else None,
                'epoch': epoch,
                'dev_eer': dev_eer,
                'args': vars(args),
            }
            torch.save(best_state, out_dir / "best.pt")

    # Load best and evaluate on dev/test; save artifacts
    if best_state is None:
        print("No improvement recorded; saving last state.")
        best_state = {
            'model': model.state_dict(),
            'clf': clf.state_dict() if clf is not None else None,
            'epoch': args.epochs,
            'dev_eer': float('nan'),
            'args': vars(args),
        }
        torch.save(best_state, out_dir / "best.pt")
    else:
        model.load_state_dict(best_state['model'])

    evaluate_and_save(model, dev_loader, test_loader, out_dir)
    print(f"Done. Artifacts in: {out_dir}")

def evaluate(model, loader) -> float:
    device = next(model.parameters()).device
    embeds, labels = embed_loader(model, loader, device)
    y_true, y_score = mean_enroll(embeds, labels, per_subject=2)
    if len(y_true) == 0:
        return float('nan')
    res = compute_eer(y_true, y_score)
    return res['eer']

def evaluate_and_save(model, dev_loader, test_loader, out_dir: Path):
    device = next(model.parameters()).device
    # Dev
    dev_emb, dev_y = embed_loader(model, dev_loader, device)
    y_true_d, y_score_d = mean_enroll(dev_emb, dev_y, per_subject=2)
    dev_res = compute_eer(y_true_d, y_score_d)

    # Test
    test_emb, test_y = embed_loader(model, test_loader, device)
    y_true_t, y_score_t = mean_enroll(test_emb, test_y, per_subject=2)
    test_res = compute_eer(y_true_t, y_score_t)

    # Save scores (test)
    scores = pd.DataFrame({'label': y_true_t, 'score': y_score_t})
    scores.to_csv(out_dir / "scores.csv", index=False)

    # Save metrics
    metrics = {
        'dev_eer': dev_res['eer'],
        'dev_auc': dev_res['auc'],
        'test_eer': test_res['eer'],
        'test_auc': test_res['auc'],
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # ROC for test
    plot_roc(test_res['fpr'], test_res['tpr'], test_res['auc'], eer=test_res['eer'], out_path=out_dir / "roc.png")

    print("Metrics:", json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
