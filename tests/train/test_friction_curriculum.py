import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from train.trainer import Trainer, TrainerConfig


def test_friction_curriculum_transitions():
    cfg = TrainerConfig(
        max_steps=1000,
        friction_smooth_schedule=True,
        friction_smooth_fraction=0.3,
    )
    tr = Trainer(cfg)
    s1 = tr._resolve_friction_mode_for_step(50)
    s2 = tr._resolve_friction_mode_for_step(250)
    s3 = tr._resolve_friction_mode_for_step(900)
    assert s1 == "off"
    assert s2 in {"smooth", "blend"}
    assert s3 == "strict"


def test_friction_curriculum_applies_in_residual_mode():
    cfg = TrainerConfig(
        max_steps=1000,
        friction_smooth_schedule=True,
        friction_smooth_fraction=0.3,
    )
    cfg.total_cfg.loss_mode = "residual"
    tr = Trainer(cfg)
    assert tr._resolve_friction_mode_for_step(50) == "off"
