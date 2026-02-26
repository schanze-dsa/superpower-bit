import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from train.trainer import TrainerConfig


def test_viz_surface_source_defaults_to_surface():
    cfg = TrainerConfig()
    assert cfg.viz_surface_source == "surface"
