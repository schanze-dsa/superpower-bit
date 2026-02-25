import sys
from pathlib import Path

import pytest
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from model.loss_energy import TotalConfig, TotalEnergy


def test_staged_preload_uses_delta_not_repeated_total():
    cfg = TotalConfig(
        loss_mode="residual",
        w_int=0.0,
        w_cn=0.0,
        w_ct=0.0,
        w_fb=0.0,
        w_region=0.0,
        w_tie=0.0,
        w_bc=0.0,
        w_pre=1.0,
        w_tight=0.0,
        w_sigma=0.0,
        w_eq=0.0,
        w_reg=0.0,
    )
    total = TotalEnergy(cfg)

    def _fake_parts(_u_fn, params, _tape=None, stress_fn=None):  # noqa: ARG001
        return {"W_pre": tf.cast(params["w_pre"], tf.float32)}, {}

    total._compute_parts_residual = _fake_parts  # type: ignore[method-assign]

    stages = [
        {"P": tf.constant([1.0], dtype=tf.float32), "P_hat": tf.constant([0.0], dtype=tf.float32), "w_pre": 10.0},
        {"P": tf.constant([1.0], dtype=tf.float32), "P_hat": tf.constant([0.0], dtype=tf.float32), "w_pre": 30.0},
    ]
    pi, _parts, stats = total._residual_staged(lambda x, params=None: x, stages, root_params={})

    assert float(pi.numpy()) == pytest.approx(30.0)
    assert "s2_delta_W_pre" in stats
    assert float(tf.convert_to_tensor(stats["s2_delta_W_pre"]).numpy()) == pytest.approx(20.0)
