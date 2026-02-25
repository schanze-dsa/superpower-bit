import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from train.trainer import Trainer, TrainerConfig


def test_order_bank_covers_all_permutations_for_3_bolts():
    cfg = TrainerConfig(preload_use_stages=True, seed=7)
    tr = Trainer(cfg)
    tr._preload_dim = 3
    seen = set()
    for _ in range(24):
        case = tr._sample_preload_case()
        order = tuple(int(x) for x in case["order"].tolist())
        seen.add(order)
    assert len(seen) == 6
