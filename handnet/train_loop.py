from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from .losses import batch_hard_triplet_loss


def train_one_epoch(model, clf: Optional[torch.nn.Module], loader, optimizer, device, loss_type="ce"):
    model.train()
    if clf is not None:
        clf.train()
    losses = []
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()
        z = model(xb)
        if loss_type == "ce":
            logits = clf(z)
            loss = F.cross_entropy(logits, yb)
        else:
            loss = batch_hard_triplet_loss(z, yb, margin=0.2)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else 0.0

