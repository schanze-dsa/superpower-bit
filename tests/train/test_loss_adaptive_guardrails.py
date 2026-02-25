import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from train.loss_weights import LossWeightState, update_loss_weights


def test_non_focus_terms_not_overamplified():
    state = LossWeightState.init(
        base_weights={"E_cn": 1.0, "E_eq": 1.0, "W_pre": 1.0},
        focus_terms=("E_cn", "E_eq"),
        min_weight=1e-4,
        max_weight=10.0,
    )
    parts = {"E_cn": 100.0, "E_eq": 50.0, "W_pre": 1e6}
    update_loss_weights(state, parts)
    assert state.current["W_pre"] == 1.0
