import sys
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from physics.tightening_model import NutSampleData, NutTighteningPenalty, TighteningConfig


def _build_penalty(X: np.ndarray, w: np.ndarray) -> NutTighteningPenalty:
    model = NutTighteningPenalty(TighteningConfig(alpha=1e3, angle_unit="rad", clockwise=False))
    model._nuts = [
        NutSampleData(
            name="n1",
            X=X.astype(np.float32),
            w=w.astype(np.float32),
            tri_node_idx=None,
            bary=None,
            axis=np.array([0.0, 0.0, 1.0], dtype=np.float32),
            center=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        )
    ]
    return model


def test_tightening_residual_normalized_by_total_weight():
    X = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    w = np.array([1.0, 1.0], dtype=np.float32)
    model_a = _build_penalty(X, w)

    X_b = np.repeat(X, 2, axis=0)
    w_b = np.repeat(w, 2, axis=0)
    model_b = _build_penalty(X_b, w_b)

    params = {"theta": tf.constant([0.1], dtype=tf.float32)}
    zero_u = lambda x, p: tf.zeros_like(x)

    res_a, stats_a = model_a.residual(zero_u, params)
    res_b, _stats_b = model_b.residual(zero_u, params)

    assert float(res_a.numpy()) == pytest.approx(float(res_b.numpy()), rel=1e-6)
    assert "weight_sum" in stats_a["tightening"]
    assert "rms_per_nut" in stats_a["tightening"]
