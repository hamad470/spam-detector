"""Group near-duplicate messages into campaign clusters.

Spam is sent from templates: same wording, different phone number or prize.
If one copy lands in train and another in test, the test score mostly measures
memorisation. Grouping lets the split keep each campaign on one side.
"""

import re

from datasketch import MinHash, MinHashLSH


def _signature(text: str, num_perm: int, k: int) -> MinHash:
    # Digits collapse to 0 so "call 09061701444" and "call 09061701461" share shingles
    t = re.sub(r"\d", "0", re.sub(r"\s+", " ", text.lower()))
    h = MinHash(num_perm=num_perm)
    for i in range(max(1, len(t) - k + 1)):
        h.update(t[i : i + k].encode())
    return h


def near_duplicate_groups(texts, threshold: float = 0.7, num_perm: int = 128, k: int = 5) -> list[int]:
    """Return a cluster id per text; texts with estimated Jaccard >= threshold share an id."""
    texts = list(texts)
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    sigs = []
    for i, t in enumerate(texts):
        sigs.append(_signature(t, num_perm, k))
        lsh.insert(i, sigs[-1])

    parent = list(range(len(texts)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, sig in enumerate(sigs):
        for j in lsh.query(sig):
            parent[find(i)] = find(j)
    return [find(i) for i in range(len(texts))]
