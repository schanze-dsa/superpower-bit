from pathlib import Path

import yaml


def test_physics_only_smoke_config_has_required_keys():
    cfg_path = Path("config.yaml")
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}

    smoke = (data.get("integration_smoke") or {}).get("physics_only") or {}
    required = [
        "preload_use_stages",
        "incremental_mode",
        "loss_config",
        "friction_config",
        "seed",
        "viz_plot_stages",
    ]
    for key in required:
        assert key in smoke
