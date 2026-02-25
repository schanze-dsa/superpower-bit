# PINN Bolt-Tightening Mirror Deformation Stabilization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a stable physics-only PINN pipeline that supports arbitrary tightening order, exports stage and final mirror cloud maps, and converges reliably without supervised truth labels.

**Architecture:** Keep the existing staged PINN + DFEM/contact framework, but add strict order-coverage sampling, staged friction curriculum, guarded adaptive losses, and deterministic visualization exports. Implement changes incrementally with TDD and small commits.

**Tech Stack:** Python, TensorFlow, NumPy, PyTest, YAML config, existing project modules under `src/`.

---

### Task 1: Add tests for staged order feature encoding

**Files:**
- Create: `tests/train/test_stage_order_features.py`
- Modify: `src/train/trainer.py`
- Test: `tests/train/test_stage_order_features.py`

**Step 1: Write the failing test**

```python
import numpy as np

from train.trainer import TrainerConfig, Trainer


def test_stage_features_shape_and_rank_encoding():
    cfg = TrainerConfig(preload_use_stages=True)
    tr = Trainer(cfg)
    tr._preload_dim = 3
    p = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    order = np.array([2, 0, 1], dtype=np.int32)

    stage_p, stage_feat = tr._build_stage_tensors(p, order)

    assert stage_p.shape[0] >= 3
    assert stage_feat.shape[1] == 12  # 4 * n_bolts for n_bolts=3
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/train/test_stage_order_features.py -q`  
Expected: FAIL due to missing/incorrect shape or feature construction.

**Step 3: Write minimal implementation**

```python
# In Trainer._build_stage_tensors:
# - ensure output feature layout [normP, mask, last, rank]
# - ensure feature dim is exactly 4 * n_bolts
# - keep rank normalized to [0,1]
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/train/test_stage_order_features.py -q`  
Expected: PASS

**Step 5: Commit**

```bash
git add tests/train/test_stage_order_features.py src/train/trainer.py
git commit -m "test(train): validate staged order feature encoding"
```

### Task 2: Implement balanced order-coverage sampler

**Files:**
- Create: `tests/train/test_order_bank_sampler.py`
- Modify: `src/train/trainer.py`
- Test: `tests/train/test_order_bank_sampler.py`

**Step 1: Write the failing test**

```python
import numpy as np

from train.trainer import TrainerConfig, Trainer


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
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/train/test_order_bank_sampler.py -q`  
Expected: FAIL because current sampling is not guaranteed to cover all permutations.

**Step 3: Write minimal implementation**

```python
# In Trainer:
# - add an order-bank queue for permutations
# - refill with shuffled full permutation set when empty
# - consume one order per sampled case
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/train/test_order_bank_sampler.py -q`  
Expected: PASS

**Step 5: Commit**

```bash
git add tests/train/test_order_bank_sampler.py src/train/trainer.py
git commit -m "feat(train): guarantee order permutation coverage"
```

### Task 3: Guard adaptive loss terms for physics-only mode

**Files:**
- Create: `tests/train/test_loss_adaptive_guardrails.py`
- Modify: `src/train/loss_weights.py`
- Modify: `src/model/loss_energy.py`
- Test: `tests/train/test_loss_adaptive_guardrails.py`

**Step 1: Write the failing test**

```python
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
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/train/test_loss_adaptive_guardrails.py -q`  
Expected: FAIL if non-focus terms drift.

**Step 3: Write minimal implementation**

```python
# In update_loss_weights:
# - apply adaptive updates only on focus_terms
# - keep other terms fixed at base/current value
# - enforce min/max clipping after each update
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/train/test_loss_adaptive_guardrails.py -q`  
Expected: PASS

**Step 5: Commit**

```bash
git add tests/train/test_loss_adaptive_guardrails.py src/train/loss_weights.py src/model/loss_energy.py
git commit -m "fix(loss): constrain adaptive weighting to focus terms"
```

### Task 4: Add friction curriculum scheduler (no-friction -> smooth -> strict)

**Files:**
- Create: `tests/train/test_friction_curriculum.py`
- Modify: `src/train/trainer.py`
- Modify: `main.py`
- Test: `tests/train/test_friction_curriculum.py`

**Step 1: Write the failing test**

```python
from train.trainer import TrainerConfig, Trainer


def test_friction_curriculum_transitions():
    cfg = TrainerConfig(max_steps=1000, friction_smooth_schedule=True, friction_smooth_fraction=0.3)
    tr = Trainer(cfg)
    s1 = tr._resolve_friction_mode_for_step(50)
    s2 = tr._resolve_friction_mode_for_step(250)
    s3 = tr._resolve_friction_mode_for_step(900)
    assert s1 == "off"
    assert s2 in {"smooth", "blend"}
    assert s3 == "strict"
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/train/test_friction_curriculum.py -q`  
Expected: FAIL because helper/schedule does not exist or behavior differs.

**Step 3: Write minimal implementation**

```python
# In Trainer:
# - add _resolve_friction_mode_for_step(step)
# - apply mode at step boundaries to contact/friction config
# In main.py:
# - parse explicit curriculum window config keys
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/train/test_friction_curriculum.py -q`  
Expected: PASS

**Step 5: Commit**

```bash
git add tests/train/test_friction_curriculum.py src/train/trainer.py main.py
git commit -m "feat(train): add staged friction curriculum scheduler"
```

### Task 5: Enforce preload delta accounting in staged loss

**Files:**
- Create: `tests/model/test_staged_preload_delta.py`
- Modify: `src/model/loss_energy.py`
- Test: `tests/model/test_staged_preload_delta.py`

**Step 1: Write the failing test**

```python
import tensorflow as tf

from model.loss_energy import TotalEnergy, TotalConfig


def test_staged_preload_uses_delta_not_repeated_total():
    total = TotalEnergy(TotalConfig())
    # Build minimal staged inputs and verify Pi accumulation uses delta_W_pre.
    # Expected: cumulative W_pre is not double-counted across stages.
    assert True
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/model/test_staged_preload_delta.py -q`  
Expected: FAIL with assertion or missing fixture logic.

**Step 3: Write minimal implementation**

```python
# In _energy_staged and _residual_staged:
# - track prev_W_pre
# - apply delta_W_pre in step objective
# - keep stats explicit: sN_delta_W_pre, sN_Pi_step, path penalties
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/model/test_staged_preload_delta.py -q`  
Expected: PASS

**Step 5: Commit**

```bash
git add tests/model/test_staged_preload_delta.py src/model/loss_energy.py
git commit -m "fix(loss): use preload increments in staged objective"
```

### Task 6: Normalize tightening residual scale across bolts/stages

**Files:**
- Create: `tests/physics/test_tightening_residual_scale.py`
- Modify: `src/physics/tightening_model.py`
- Test: `tests/physics/test_tightening_residual_scale.py`

**Step 1: Write the failing test**

```python
import tensorflow as tf

from physics.tightening_model import NutTighteningPenalty, TighteningConfig


def test_tightening_residual_normalized_by_total_weight():
    model = NutTighteningPenalty(TighteningConfig(alpha=1e3))
    # Build synthetic nuts and verify residual remains comparable when sample count changes.
    assert True
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/physics/test_tightening_residual_scale.py -q`  
Expected: FAIL (placeholder behavior or no normalization guarantee).

**Step 3: Write minimal implementation**

```python
# In residual():
# - normalize by effective total area/weight (already partly present)
# - verify consistency for varying point counts per nut
# - keep returned stats for per-nut RMS
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/physics/test_tightening_residual_scale.py -q`  
Expected: PASS

**Step 5: Commit**

```bash
git add tests/physics/test_tightening_residual_scale.py src/physics/tightening_model.py
git commit -m "fix(physics): stabilize tightening residual scaling"
```

### Task 7: Add deterministic stage/final mirror export checks

**Files:**
- Create: `tests/viz/test_stage_final_exports.py`
- Modify: `src/viz/mirror_viz.py`
- Modify: `src/train/trainer.py`
- Test: `tests/viz/test_stage_final_exports.py`

**Step 1: Write the failing test**

```python
from pathlib import Path


def test_stage_and_final_exports_have_expected_names(tmp_path: Path):
    # Verify file naming convention includes stage/final and order metadata.
    expected = ["mirror_stage_01.png", "mirror_stage_02.png", "mirror_final.png"]
    for name in expected:
        assert isinstance(name, str)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/viz/test_stage_final_exports.py -q`  
Expected: FAIL because naming/export contract is not yet enforced.

**Step 3: Write minimal implementation**

```python
# In trainer + viz:
# - enforce deterministic output naming and metadata labels
# - export stage maps when viz_plot_stages=true
# - always export final map and paired text data
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/viz/test_stage_final_exports.py -q`  
Expected: PASS

**Step 5: Commit**

```bash
git add tests/viz/test_stage_final_exports.py src/viz/mirror_viz.py src/train/trainer.py
git commit -m "feat(viz): standardize stage/final mirror deformation exports"
```

### Task 8: Run integration smoke and capture metrics

**Files:**
- Create: `tests/integration/test_physics_only_smoke.py`
- Modify: `config.yaml`
- Test: `tests/integration/test_physics_only_smoke.py`

**Step 1: Write the failing test**

```python
def test_physics_only_smoke_config_has_required_keys():
    required = [
        "preload_use_stages",
        "incremental_mode",
        "loss_config",
        "friction_config",
    ]
    for key in required:
        assert key is not None
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/integration/test_physics_only_smoke.py -q`  
Expected: FAIL until smoke fixture/config parser contract is implemented.

**Step 3: Write minimal implementation**

```python
# Add a smoke configuration profile for overnight run:
# - staged loading enabled
# - guarded adaptive terms
# - friction curriculum enabled
# - deterministic seeds
```

**Step 4: Run test and smoke command**

Run: `python -m pytest tests/integration/test_physics_only_smoke.py -q`  
Expected: PASS

Run: `python main.py`  
Expected: training starts, no immediate NaN/Inf, stage/final output files are produced.

**Step 5: Commit**

```bash
git add tests/integration/test_physics_only_smoke.py config.yaml
git commit -m "test(integration): add physics-only staged training smoke profile"
```

## Verification Checklist Before Merge

- Run: `python -m pytest tests -q`
- Run: `python main.py --export results/saved_model_smoke`
- Confirm files:
- `results/deflection_*_stage*.png`
- `results/deflection_*_final.png`
- `results/deflection_*.txt`
- Confirm logs contain no persistent NaN/Inf in loss terms.

## Notes

- Keep commits small (one task per commit).
- Prefer minimal implementation to satisfy each failing test (YAGNI).
- Avoid touching unrelated physics modules.

