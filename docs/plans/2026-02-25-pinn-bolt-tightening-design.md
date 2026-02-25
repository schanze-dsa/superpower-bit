# PINN Bolt-Tightening Mirror Deformation Design

## 1. Background

This design targets physics-only PINN prediction of mirror deformation from bolt tightening on a single CDB model.
There is no supervised displacement ground truth.
The workflow must output:

- stage-wise deformation maps during tightening
- final deformation map after full tightening

## 2. Confirmed Requirements

- No truth labels are available.
- The model must support arbitrary bolt tightening order (order generalization).
- Scope is a single CDB model; generalization is over load magnitudes and order.
- Training should follow a stability-first curriculum: no friction first, then friction.
- Budget is a single GPU overnight run (roughly 8-24 hours).
- Output must be reproducible and directly usable for mirror cloud-map reporting.

## 3. Candidate Approaches

### Approach A (Recommended): Sequence-conditioned staged physics PINN

Use one displacement network conditioned by:

- preload magnitudes
- staged masks/last-active flags
- tightening order rank features

Run staged physics loss accumulation and generate both stage and final maps.

Pros:

- Best fit to current codebase and existing staged pipeline.
- Balanced implementation risk and training stability.
- Directly supports required outputs.

Cons:

- Sensitive to loss weighting and training curriculum.

### Approach B: Incremental solve with per-stage micro-optimization

Treat each stage as an incremental sub-problem and optimize with inner loops before moving to next stage.

Pros:

- Very stable for contact-heavy runs.

Cons:

- Slower and more complex runtime orchestration.

### Approach C: Multi-fidelity global-local two-model system

Use a coarse global model plus local contact-zone refinement model.

Pros:

- Potentially higher local accuracy and faster inference after tuning.

Cons:

- Highest complexity and integration cost.

## 4. Chosen Design

Approach A is selected, with a limited incremental strategy from Approach B.

### 4.1 Architecture

- Input: spatial coordinate `x` + staged preload/order features `P_hat`.
- `P_hat` layout remains explicit and stable: normalized preload, mask, last-active, rank.
- Backbone: global displacement branch plus contact-gated local correction branch.
- Physics assembly remains in `TotalEnergy`, with staged path handling.

### 4.2 Components to Upgrade

- staged case sampler: enforce order coverage across all permutations
- training curriculum scheduler: no-friction to smooth-friction to strict-friction
- adaptive weight guardrails: restrict terms that may drift in physics-only training
- staged energy accumulation: strict delta handling for preload work
- tightening residual normalization: keep scale stable across stage counts
- visualization exports: deterministic stage/final outputs and metadata

### 4.3 Data Flow

1. Parse CDB and build mesh/contact/tightening structures.
2. Sample staged preload cases with order-conditioned features.
3. Forward PINN displacement for mesh/contact/tightening points.
4. Compute physics terms and combine with guarded weights.
5. Update ALM/contact states on configured cadence.
6. Export per-stage and final mirror deformation maps and text data.

## 5. Training Curriculum

- Phase A (early): no friction, stabilize contact normal and boundary consistency.
- Phase B (middle): smooth friction enabled with gradual contact hardening.
- Phase C (late): strict friction and final residual polishing.

Key constraints:

- `w_int` cannot stay at zero.
- preload work and path penalties must not be freely amplified by adaptive weighting.
- order sampling must be balanced, not only random.

## 6. Error Handling and Stability Rules

- clip adaptive weight range (`min_weight`, `max_weight`) and update factor range.
- keep gradient clipping enabled.
- guard against invalid stage tensors and incompatible schedule settings.
- fail fast on missing required config keys or illegal ranges.
- preserve deterministic seeds for sampler and contact resampling.

## 7. Validation Plan (Physics-Only)

Without labels, acceptance uses physics closure and stability metrics:

- no NaN/Inf through full training run
- penetration and complementarity residual trend down
- equilibrium and boundary residual trend down
- staged-to-final field consistency
- reproducible stage/final mirror cloud-map exports

## 8. Deliverables

- updated training and loss scheduling pipeline
- balanced order-generalization sampler
- stage/final mirror deformation export package
- ablation-ready config variants and run logs

## 9. Out of Scope

- cross-geometry model generalization
- supervised displacement fitting
- multi-model domain adaptation

