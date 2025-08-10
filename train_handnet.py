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
  --backbone resnet18 --loss ce --aspect palmar --epochs 15 --batch_size 64 --img_size 224 --device auto \
  --exp_root experiments --exp_name palmar_clean --stage A_ce

# Train custom CNN with triplet loss on dorsal hands (P×K)
python train_handnet.py --data_dir ~/hands/images --csv_path ~/hands/HandInfo.csv \
  --backbone handnet --loss triplet --aspect dorsal --epochs 25 --img_size 224 --device auto \
  --pk 16x4 --exp_root experiments --exp_name dorsal_clean --stage B_triplet_pk16x4

# Evaluate only (given weights file)
python train_handnet.py --data_dir ~/hands/images --csv_path ~/hands/HandInfo.csv \
  --backbone resnet18 --eval_only --weights path/to/best.pt --aspect palmar --img_size 224 --device auto \
  --exp_root experiments --exp_name palmar_clean --stage eval_only
"""
import argparse
import time
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from handnet import (
    find_device,
    set_seed,
    ensure_min_images_per_subject,
    filter_by_aspect,
    filter_clean,
    stratified_subject_split,
    build_image_index,
    HandDataset,
    make_transforms,
    make_backbone,
    LinearClassifier,
    train_one_epoch,
    evaluate,
    evaluate_and_save,
    BalancedPKSampler
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--csv_path", required=True)
    p.add_argument("--aspect", default="palmar", choices=["palmar", "dorsal", "any"])
    p.add_argument("--clean_only", action="store_true", help="Drop accessories/nailPolish/irregularities if present")
    p.add_argument("--min_images", type=int, default=3)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--backbone", default="resnet18", choices=["resnet18", "handnet"])
    p.add_argument("--loss", default="ce", choices=["ce", "triplet"])
    p.add_argument("--device", default="auto", help="auto|cpu|cuda|mps")
    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--weights", default="", help="Path to weights for eval_only")
    p.add_argument("--init_from", default="", help="Initialize backbone weights from a prior best.pt")
    p.add_argument("--pk", default="", help="Use P×K sampler for triplet, e.g. 16x4")
    # Experiment foldering
    p.add_argument("--exp_root", default="experiments")
    p.add_argument("--exp_name", default="palmar_clean")     # e.g. palmar_clean, palmar_clean_cropped
    p.add_argument("--stage",    default="A_ce")             # e.g. A_ce, B_triplet_pk16x4
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
    train_sub, dev_sub, test_sub = stratified_subject_split(subjects, ratios=(0.7, 0.15, 0.15), seed=args.seed)
    df_train = df[df['id'].isin(train_sub)].copy()
    df_dev   = df[df['id'].isin(dev_sub)].copy()
    df_test  = df[df['id'].isin(test_sub)].copy()
    print(f"Split → train/dev/test subjects: {len(train_sub)}/{len(dev_sub)}/{len(test_sub)}")

    # Transforms
    train_tf, eval_tf = make_transforms(args.img_size, aug=True)

    # Label remap for classifier (train-only)
    id_to_idx = {sid: i for i, sid in enumerate(sorted(df_train['id'].unique()))}
    def remap(df_in):
        return df_in.assign(lbl=df_in['id'].map(id_to_idx).fillna(-1).astype(int))

    df_train = remap(df_train)
    df_dev   = remap(df_dev)
    df_test  = remap(df_test)

    # Datasets
    if args.loss == "ce":
        # IMPORTANT: use remapped labels for CE training
        train_ds = HandDataset(df_train[['imageName', 'lbl']].rename(columns={'lbl': 'id'}),
                               img_index, transform=train_tf)
    else:
        train_ds = HandDataset(df_train[['imageName', 'id']], img_index, transform=train_tf)

    dev_ds  = HandDataset(df_dev[['imageName', 'id']], img_index, transform=eval_tf)
    test_ds = HandDataset(df_test[['imageName', 'id']], img_index, transform=eval_tf)

    # Loaders (P×K sampler for triplet if requested)
    if args.loss == "triplet" and args.pk:
        P, K = map(int, args.pk.lower().replace('x', ' ').split())
        train_labels = df_train['id'].tolist()
        sampler = BalancedPKSampler(train_labels, P=P, K=K, seed=args.seed)
        train_loader = DataLoader(train_ds, batch_sampler=sampler, num_workers=0)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)

    dev_loader  = DataLoader(dev_ds,  batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Model
    embed_dim = 128
    model = make_backbone(args.backbone, embed_dim=embed_dim).to(device)

    # Optionally initialize from prior best.pt (backbone only)
    if args.init_from:
        state = torch.load(args.init_from, map_location=device)
        model.load_state_dict(state["model"], strict=False)
        print(f"Initialized backbone from {args.init_from}")

    # CE classifier head (train-time only)
    clf = None
    if args.loss == "ce":
        clf = LinearClassifier(embed_dim, num_classes=len(id_to_idx)).to(device)
        params = list(model.parameters()) + list(clf.parameters())
    else:
        params = list(model.parameters())

    # Optimizer
    opt = torch.optim.Adam(params, lr=1e-3, weight_decay=1e-4)

    # Experiment out_dir: experiments/<exp_name>/<backbone>/<stage>/<timestamp_seed>/
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_leaf = f"{timestamp}_seed{args.seed}"
    out_dir = Path(args.exp_root) / args.exp_name / args.backbone / args.stage / run_leaf
    out_dir.mkdir(parents=True, exist_ok=True)

    # Eval-only
    if args.eval_only:
        assert args.weights, "--weights required for --eval_only"
        state = torch.load(args.weights, map_location=device)
        model.load_state_dict(state['model'], strict=False)
        if 'clf' in state and clf is not None and state['clf'] is not None:
            clf.load_state_dict(state['clf'])
        # Save args alongside metrics for bookkeeping
        with open(out_dir / "args.json", "w") as f:
            json.dump(vars(args), f, indent=2)
        evaluate_and_save(model, dev_loader, test_loader, out_dir)
        print(f"Done. Artifacts in: {out_dir}")
        return

    # Train loop
    best_eer = 1.0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        loss_val = train_one_epoch(model, clf, train_loader, opt, device, loss_type=args.loss)
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

    # Save args and run final eval
    with open(out_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    evaluate_and_save(model, dev_loader, test_loader, out_dir)
    print(f"Done. Artifacts in: {out_dir}")


if __name__ == "__main__":
    main()
