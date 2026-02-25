import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from train.trainer import Trainer, TrainerConfig


def test_stage_features_shape_and_rank_encoding():
    cfg = TrainerConfig(preload_use_stages=True)
    tr = Trainer(cfg)
    tr._preload_dim = 3
    p = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    order = np.array([2, 0, 1], dtype=np.int32)

    stage_p, stage_feat = tr._build_stage_tensors(p, order)

    assert stage_p.shape[0] >= 3
    assert stage_feat.shape[1] == 12  # 4 * n_bolts for n_bolts=3
