import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score


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
    """
    Build enroll mean per subject from the first `per_subject` samples,
    then score each probe against its own enroll mean (genuine) and the
    hardest *other* subject's enroll mean (impostor).
    """
    # group indices by subject id
    by_label = {}
    for i, y in enumerate(labels.tolist()):
        by_label.setdefault(y, []).append(i)

    enroll_means, enroll_labels, probes = [], [], []
    for y, idxs in by_label.items():
        if len(idxs) < per_subject + 1:
            continue
        e_idx = idxs[:per_subject]
        p_idx = idxs[per_subject:]
        enroll_means.append(embeds[e_idx].mean(0, keepdim=True))
        enroll_labels.append(y)
        for j in p_idx:
            probes.append((j, y))

    if len(enroll_means) < 2 or not probes:
        return np.array([]), np.array([])

    enroll_means  = torch.cat(enroll_means, 0)      # (S,D)
    enroll_labels = torch.tensor(enroll_labels)     # (S,)

    genu_s, imp_s = [], []
    for j, y in probes:
        p = embeds[j].unsqueeze(0)                  # (1,D)
        own = (enroll_labels == y)
        # genuine vs own mean
        e_own = enroll_means[own][0].unsqueeze(0)
        gs = F.cosine_similarity(e_own, p).item()
        genu_s.append(gs)
        # impostor = max sim among other subjects
        other = ~own
        sims = F.cosine_similarity(enroll_means[other], p.expand_as(enroll_means[other]))
        imp_s.append(float(sims.max().item()))

    y_true  = np.array([1]*len(genu_s) + [0]*len(imp_s))
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
    plt.plot([0, 1], [0, 1], linestyle='--', label='Chance')
    if eer is not None:
        diffs = np.abs((1 - tpr) - fpr)
        j = int(np.nanargmin(diffs))
        plt.scatter([fpr[j]], [tpr[j]], marker='o', label=f'EER≈{eer*100:.1f}%')
    plt.xlabel('False Positive Rate (FMR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()


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
    dev_emb, dev_y = embed_loader(model, dev_loader, device)
    y_true_d, y_score_d = mean_enroll(dev_emb, dev_y, per_subject=2)
    dev_res = compute_eer(y_true_d, y_score_d)

    test_emb, test_y = embed_loader(model, test_loader, device)
    y_true_t, y_score_t = mean_enroll(test_emb, test_y, per_subject=2)
    test_res = compute_eer(y_true_t, y_score_t)

    scores = pd.DataFrame({'label': y_true_t, 'score': y_score_t})
    scores.to_csv(out_dir / "scores.csv", index=False)

    metrics = {
        'dev_eer': dev_res['eer'],
        'dev_auc': dev_res['auc'],
        'test_eer': test_res['eer'],
        'test_auc': test_res['auc'],
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    plot_roc(test_res['fpr'], test_res['tpr'], test_res['auc'], eer=test_res['eer'], out_path=out_dir / "roc.png")
    print("Metrics:", json.dumps(metrics, indent=2))

