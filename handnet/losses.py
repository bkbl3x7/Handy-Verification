import torch
import torch.nn.functional as F


def batch_hard_triplet_loss(embeddings: torch.Tensor, labels: torch.Tensor, margin=0.2):
    # embeddings: (B,D), labels: (B,)
    B = embeddings.size(0)
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

