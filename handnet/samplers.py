import random
from collections import defaultdict
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

import numpy as np
from torch.utils.data import Sampler


class PKBatchSampler(Sampler[List[int]]):
    """Yields batches of indices with P identities and K samples per identity.

    If an identity has fewer than K samples, samples are repeated with replacement.
    """

    def __init__(self, labels: Sequence[int], P: int, K: int, seed: int = 42):
        self.labels: List[int] = list(map(int, labels))
        self.P = int(P)
        self.K = int(K)
        self.rng = random.Random(seed)

        self.by_label: Dict[int, List[int]] = defaultdict(list)
        for idx, y in enumerate(self.labels):
            self.by_label[y].append(idx)

        self.identities: List[int] = list(self.by_label.keys())

        # Estimate number of batches per epoch as total_samples / (P*K)
        self.batch_size = self.P * self.K
        self.num_batches = max(1, len(self.labels) // self.batch_size)

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterator[List[int]]:
        # Shuffle identities every epoch
        ids = self.identities[:]
        self.rng.shuffle(ids)

        # Draw batches
        i = 0
        for _ in range(self.num_batches):
            # sample P identities (with replacement if needed)
            if i + self.P > len(ids):
                # reshuffle for the remainder
                self.rng.shuffle(ids)
                i = 0
            chosen = ids[i:i + self.P]
            if len(chosen) < self.P:
                # pad with random choices if not enough identities
                chosen += self.rng.choices(ids, k=self.P - len(chosen))
            i += self.P

            batch: List[int] = []
            for y in chosen:
                idxs = self.by_label[y]
                if len(idxs) >= self.K:
                    batch.extend(self.rng.sample(idxs, k=self.K))
                else:
                    # sample with replacement
                    batch.extend(self.rng.choices(idxs, k=self.K))
            # final shuffle within batch for randomness
            self.rng.shuffle(batch)
            yield batch

