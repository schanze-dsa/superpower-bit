#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
loss_energy.py
--------------
Total potential energy assembly for DFEM/PINN with contact & preloads.

Composition (                                                 loss_weights       )  ?       = w_int * E_int
        + w_cn * E_cn
        + w_ct * E_ct
        + w_tie * E_tie
        - w_pre * W_pre
        + w_sigma * E_sigma
        + w_eq * E_eq

Public usage (typical):
    # 1) Build sub-operators per batch
    elas.build_from_numpy(...) / build_dfem_subcells(...)
    contact.build_from_cat(cat_dict, extra_weights=..., auto_orient=True)
    tie.build_from_numpy(xs, xm, w_area, dof_mask=None)
    # (     ?                                       ?BoundaryPenalty         ?bcs  ?    #                                                ?
    # 2) Assemble total energy
    total = TotalEnergy()
    total.attach(elasticity=elas, contact=contact, preload=preload, ties=[tie], bcs=[bc])

    # 3) Compute energy & update multipliers in training loop
    Pi, parts, stats = total.energy(model.u_fn, params={"P": [P1,P2,P3]})
    #                                             ?loss_weights.update_loss_weights / combine_loss
    if step % total.cfg.update_every_steps == 0:
        total.update_multipliers(model.u_fn, params)

Weighted PINN:
    - You can multiply extra per-sample weights into components:
        contact.multiply_weights(w_contact)
        for t in ties: t.multiply_weights(w_tie)
        #                                               BoundaryPenalty                                 ?    - If you need to reweight volume points, see TotalEnergy.scale_volume_weights().
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import numpy as np
import tensorflow as tf

# sub-operators
from physics.elasticity_energy import ElasticityEnergy, ElasticityConfig
from physics.contact.contact_operator import ContactOperator, ContactOperatorConfig
from physics.tie_constraints import TiePenalty, TieConfig
from physics.boundary_conditions import BoundaryPenalty, BoundaryConfig
from physics.preload_model import PreloadWork, PreloadConfig
from physics.tightening_model import NutTighteningPenalty, TighteningConfig


# -----------------------------
# Config for total energy
# -----------------------------

@dataclass
class TotalConfig:
    # coefficients for each term (                    ?loss_weights             base_weights)
    loss_mode: str = "energy"  # "energy" | "residual"
    w_int: float = 1.0
    w_cn: float = 1.0            # normal contact  -> E_cn
    w_ct: float = 1.0            # frictional      -> E_ct
    w_fb: float = 0.0            # complementarity residual (Fischer-Burmeister)
    w_region: float = 0.0        # contact-region curriculum penalty
    w_tie: float = 1.0
    w_bc: float = 1.0            # boundary penalty -> E_bc
    w_pre: float = 1.0           # multiplies the subtracted W_pre
    w_tight: float = 1.0         # nut tightening rotation penalty
    w_sigma: float = 1.0         # stress supervision term (  _pred vs   _phys)
    w_eq: float = 0.0            # equilibrium residual term (div   _pred   ?0)
    w_reg: float = 0.0           # regularization term (residual-only mode)
    #                              ?E_sigma                                                   ?MPa  ?    sigma_ref: float = 1.0

    # staged preload                                                             
    path_penalty_weight: float = 1.0
    fric_path_penalty_weight: float = 1.0

    # Region curriculum (for contact-focused residuals)
    region_curriculum_start: float = 0.2
    region_curriculum_end: float = 0.8
    region_focus_power: float = 2.0
    region_focus_sigma: float = 1.0

    # staged preload mode:
    # - "cumulative_force": existing behaviour (forces remain applied cumulatively)
    # - "force_then_lock": only current bolt is force-controlled; earlier bolts are locked
    preload_stage_mode: str = "cumulative_force"

    adaptive_scheme: str = "contact_only"

    # ALM outer update cadence for contact (can be used by training loop)
    update_every_steps: int = 150

    # dtype
    dtype: str = "float32"


# -----------------------------
# Total energy assembler
# -----------------------------

class TotalEnergy:
    """
    Assemble total potential energy from provided operators.

    - energy(...)                               /                    ?          _total, parts_dict, stats_dict
             parts_dict               ?train/loss_weights.py                     ?
    - update_multipliers(...)               ?ALM                     ?           ?    """

    def __init__(self, cfg: Optional[TotalConfig] = None):
        self.cfg = cfg or TotalConfig()
        self.dtype = tf.float32 if self.cfg.dtype == "float32" else tf.float64

        # sub-ops (optional ones can be None)
        self.elasticity: Optional[ElasticityEnergy] = None
        self.contact: Optional[ContactOperator] = None
        self.ties: List[TiePenalty] = []
        self.bcs: List[BoundaryPenalty] = []
        self.preload: Optional[PreloadWork] = None
        self.tightening: Optional[NutTighteningPenalty] = None

        # trainable (non) scalars as TF vars so they can be scheduled
        self._ensure_weight_vars()

        self._built = False

    def _ensure_weight_vars(self):
        """                                      ?checkpoint/                                ?"""

        if not hasattr(self, "w_int"):
            self.w_int = tf.Variable(self.cfg.w_int, dtype=self.dtype, trainable=False, name="w_int")
        if not hasattr(self, "w_cn"):
            self.w_cn = tf.Variable(self.cfg.w_cn, dtype=self.dtype, trainable=False, name="w_cn")
        if not hasattr(self, "w_ct"):
            self.w_ct = tf.Variable(self.cfg.w_ct, dtype=self.dtype, trainable=False, name="w_ct")
        if not hasattr(self, "w_fb"):
            self.w_fb = tf.Variable(self.cfg.w_fb, dtype=self.dtype, trainable=False, name="w_fb")
        if not hasattr(self, "w_region"):
            self.w_region = tf.Variable(
                self.cfg.w_region, dtype=self.dtype, trainable=False, name="w_region"
            )
        if not hasattr(self, "w_tie"):
            self.w_tie = tf.Variable(self.cfg.w_tie, dtype=self.dtype, trainable=False, name="w_tie")
        if not hasattr(self, "w_bc"):
            self.w_bc = tf.Variable(self.cfg.w_bc, dtype=self.dtype, trainable=False, name="w_bc")
        if not hasattr(self, "w_pre"):
            self.w_pre = tf.Variable(self.cfg.w_pre, dtype=self.dtype, trainable=False, name="w_pre")
        if not hasattr(self, "w_tight"):
            self.w_tight = tf.Variable(self.cfg.w_tight, dtype=self.dtype, trainable=False, name="w_tight")
        if not hasattr(self, "w_sigma"):
            self.w_sigma = tf.Variable(self.cfg.w_sigma, dtype=self.dtype, trainable=False, name="w_sigma")
        if not hasattr(self, "w_eq"):
            self.w_eq = tf.Variable(self.cfg.w_eq, dtype=self.dtype, trainable=False, name="w_eq")
        if not hasattr(self, "w_reg"):
            self.w_reg = tf.Variable(self.cfg.w_reg, dtype=self.dtype, trainable=False, name="w_reg")

    def _loss_mode(self) -> str:
        mode = str(getattr(self.cfg, "loss_mode", "energy") or "energy").strip().lower()
        if mode in {"residual", "residual_only", "res"}:
            return "residual"
        return "energy"

    def _region_curriculum_coeff(self, params) -> tf.Tensor:
        """Smoothstep curriculum coefficient in [0,1] based on training progress."""

        dtype = self.dtype
        start = float(getattr(self.cfg, "region_curriculum_start", 0.2))
        end = float(getattr(self.cfg, "region_curriculum_end", 0.8))
        if end <= start:
            return tf.cast(1.0, dtype)

        progress = None
        if isinstance(params, dict):
            if "train_progress" in params:
                progress = tf.cast(params["train_progress"], dtype)
            elif "stage_fraction" in params:
                progress = tf.cast(params["stage_fraction"], dtype)

        if progress is None:
            return tf.cast(1.0, dtype)

        t = (progress - tf.cast(start, dtype)) / tf.cast(end - start, dtype)
        t = tf.clip_by_value(t, tf.cast(0.0, dtype), tf.cast(1.0, dtype))
        return t * t * (tf.cast(3.0, dtype) - tf.cast(2.0, dtype) * t)

    def _add_contact_aux_losses(self, u_fn, params, parts, stats, *, u_nodes=None):
        """Add complementarity + region curriculum terms for contact."""

        if self.contact is None or getattr(self.contact, "normal", None) is None:
            return

        dtype = self.dtype
        zero = tf.cast(0.0, dtype)
        fb_term = zero

        # Complementarity (FB) term:
        # in residual mode E_cn is already FB-like, in energy mode compute explicitly.
        if self._loss_mode() == "residual":
            fb_term = tf.cast(parts.get("E_cn", zero), dtype)
        else:
            try:
                fb_term_raw, fb_stats = self.contact.normal.residual(
                    u_fn, params, u_nodes=u_nodes
                )
                fb_term = tf.cast(fb_term_raw, dtype)
                if isinstance(fb_stats, dict):
                    for key, val in fb_stats.items():
                        if str(key).startswith("cn_"):
                            stats[f"fb_{key}"] = val
            except Exception:
                fb_term = tf.cast(parts.get("E_cn", zero), dtype)
        parts["E_fb"] = fb_term

        # Region curriculum term focused on contact neighborhood penetration.
        gap = getattr(self.contact.normal, "_last_gap", None)
        w_area = getattr(self.contact.normal, "w", None)
        region_raw = fb_term
        if gap is not None and w_area is not None:
            gap = tf.cast(gap, dtype)
            w_area = tf.cast(w_area, dtype)
            beta = tf.cast(getattr(self.contact.normal, "beta", 50.0), dtype)
            beta = tf.maximum(beta, tf.cast(1.0e-6, dtype))
            focus_sigma = tf.cast(
                max(float(getattr(self.cfg, "region_focus_sigma", 1.0)), 1.0e-6), dtype
            )
            power = tf.cast(
                max(float(getattr(self.cfg, "region_focus_power", 2.0)), 1.0), dtype
            )
            phi = tf.nn.softplus(-beta * gap / focus_sigma) / beta
            w_norm = w_area / (tf.reduce_mean(w_area) + tf.cast(1.0e-12, dtype))
            region_raw = tf.reduce_mean(w_norm * tf.pow(phi, power))

        coeff = self._region_curriculum_coeff(params)
        parts["E_region"] = tf.cast(coeff, dtype) * tf.cast(region_raw, dtype)
        stats["region_curriculum"] = coeff
        stats["region_raw"] = tf.cast(region_raw, dtype)

    # ---------- wiring ----------

    def attach(
        self,
        elasticity: Optional[ElasticityEnergy] = None,
        contact: Optional[ContactOperator] = None,
        preload: Optional[PreloadWork] = None,
        tightening: Optional[NutTighteningPenalty] = None,
        ties: Optional[List[TiePenalty]] = None,
        bcs: Optional[List[BoundaryPenalty]] = None,
    ):
        """
        Attach sub-components built for the current batch.
        """
        if elasticity is not None:
            self.elasticity = elasticity
        if contact is not None:
            self.contact = contact
        if preload is not None:
            self.preload = preload
        if tightening is not None:
            self.tightening = tightening
        if ties is not None:
            self.ties = list(ties)
        if bcs is not None:
            self.bcs = list(bcs)

        self._built = True

    def reset(self):
        """Detach everything (e.g., before building a new batch)."""
        self.elasticity = None
        self.contact = None
        self.preload = None
        self.tightening = None
        self.ties = []
        self.bcs = []
        self._built = False

    # ---------- optional helpers ----------

    def scale_volume_weights(self, factor: float):
        """
        Multiply all volume quadrature weights by 'factor' (coarse reweighting).
        Use this if you want Weighted PINN-like emphasis on volume PDE residuals.
        """
        if getattr(self.elasticity, "w_tf", None) is None:
            return
        self.elasticity.w_tf.assign(self.elasticity.w_tf * tf.cast(factor, self.dtype))

    # ---------- energy ----------

    def energy(self, u_fn, params=None, tape=None, stress_fn=None):
        """
        Compute total potential and return:
              _total, parts_dict, stats_dict

        If ``params`` contains a staged sequence (``{"stages": [...]}``) the
        energy is evaluated incrementally for each stage and accumulated so that
        different tightening orders can influence the loss.
        """
        #                                 ?checkpoint                                                                           ?        self._ensure_weight_vars()
        if not self._built:
            raise RuntimeError("[TotalEnergy] attach(...) must be called before energy().")

        if self._loss_mode() == "residual":
            if isinstance(params, dict) and params.get("stages"):
                Pi, parts, stats = self._residual_staged(
                    u_fn, params["stages"], params, tape, stress_fn=stress_fn
                )
            else:
                parts, stats = self._compute_parts_residual(u_fn, params or {}, tape, stress_fn=stress_fn)
                Pi = self._combine_parts(parts)
            return Pi, parts, stats

        if isinstance(params, dict) and params.get("stages"):
            Pi, parts, stats = self._energy_staged(
                u_fn, params["stages"], params, tape, stress_fn=stress_fn
            )
            return Pi, parts, stats

        parts, stats = self._compute_parts(u_fn, params or {}, tape, stress_fn=stress_fn)
        Pi = self._combine_parts(parts)
        return Pi, parts, stats

    def _compute_parts(self, u_fn, params, tape=None, stress_fn=None):
        """Evaluate all energy components for a given parameter dictionary."""
        dtype = self.dtype
        zero = tf.cast(0.0, dtype)
        parts: Dict[str, tf.Tensor] = {
            "E_int": zero,
            "E_cn": zero,
            "E_ct": zero,
            "E_fb": zero,
            "E_region": zero,
            "E_tie": zero,
            "E_bc": zero,
            "W_pre": zero,
            "E_tight": zero,
            "E_sigma": zero,
            "E_eq": zero,
        }
        stats: Dict[str, tf.Tensor] = {}

        u_nodes = None
        elastic_cache = None
        if self.elasticity is not None:
            u_nodes = self.elasticity._eval_u_on_nodes(u_fn, params)
            E_int_res = self.elasticity.energy(
                u_fn,
                params,
                tape=tape,
                return_cache=bool(stress_fn),
                u_nodes=u_nodes,
            )
            if bool(stress_fn):
                E_int, estates, elastic_cache = E_int_res  # type: ignore[misc]
            else:
                E_int, estates = E_int_res  # type: ignore[misc]
            parts["E_int"] = tf.cast(E_int, dtype)
            stats.update({f"el_{k}": v for k, v in estates.items()})

        if self.contact is not None:
            _, cparts, stats_cn, stats_ct = self.contact.energy(u_fn, params, u_nodes=u_nodes)
            if "E_cn" in cparts:
                parts["E_cn"] = tf.cast(cparts["E_cn"], dtype)
            elif "E_n" in cparts:
                parts["E_cn"] = tf.cast(cparts["E_n"], dtype)

            if "E_ct" in cparts:
                parts["E_ct"] = tf.cast(cparts["E_ct"], dtype)
            elif "E_t" in cparts:
                parts["E_ct"] = tf.cast(cparts["E_t"], dtype)

            stats.update(stats_cn)
            stats.update(stats_ct)

            if "R_fric_comp" in stats_ct:
                parts["R_fric_comp"] = tf.cast(stats_ct["R_fric_comp"], dtype)
            if "R_contact_comp" in stats_cn:
                parts["R_contact_comp"] = tf.cast(stats_cn["R_contact_comp"], dtype)
            self._add_contact_aux_losses(
                u_fn, params, parts, stats, u_nodes=u_nodes
            )

        # Tie constraints (always compute if present; weight can be zero at runtime)
        if self.ties:
            tie_terms = []
            for tie in self.ties:
                et, tstat = tie.energy(u_fn, params)
                tie_terms.append(tf.cast(et, dtype))
                if tstat:
                    for k, v in tstat.items():
                        stats[f"tie_{k}"] = v
            parts["E_tie"] = tf.add_n(tie_terms)

        if self.bcs:
            bc_terms = []
            for i, b in enumerate(self.bcs):
                if hasattr(b, "energy"):
                    E_bc_i, si = b.energy(u_fn, params)
                elif hasattr(b, "residual"):
                    E_bc_i, si = b.residual(u_fn, params)
                else:
                    continue
                bc_terms.append(tf.cast(E_bc_i, dtype))
                stats.update({f"bc{i+1}_{k}": v for k, v in si.items()})
            if bc_terms:
                parts["E_bc"] = tf.add_n(bc_terms)

        # Preload work (always compute if present; weight can be zero at runtime)
        if self.preload is not None:
            W_pre, pstats = self.preload.energy(u_fn, params, u_nodes=u_nodes)
            parts["W_pre"] = tf.cast(W_pre, dtype)
            #                         
            # - stats["preload"]       Trainer                           ?bolt_deltas  ?            # - stats["pre_preload"]        ?staged preload                
            stats.update(pstats)
            stats.update({f"pre_{k}": v for k, v in pstats.items()})

        if self.tightening is not None:
            E_tight, tstats = self.tightening.energy(u_fn, params, u_nodes=u_nodes)
            parts["E_tight"] = tf.cast(E_tight, dtype)
            stats.update(tstats)

        #              /                                                   ?        w_sigma = float(getattr(self.cfg, "w_sigma", 0.0))
        w_eq = float(getattr(self.cfg, "w_eq", 0.0))
        use_stress = stress_fn is not None and elastic_cache is not None
        use_sigma = use_stress and w_sigma > 1e-15 and getattr(self.elasticity.cfg, "stress_loss_weight", 0.0) > 0.0
        use_eq = use_stress and w_eq > 1e-15

        if use_sigma or use_eq:
            eps_vec = tf.cast(elastic_cache["eps_vec"], dtype)
            lam = tf.cast(elastic_cache["lam"], dtype)
            mu = tf.cast(elastic_cache["mu"], dtype)
            dof_idx = tf.cast(elastic_cache["dof_idx"], tf.int32)

            sigma_phys = elastic_cache.get("sigma_phys")
            if sigma_phys is not None:
                sigma_phys = tf.cast(sigma_phys, dtype)
            else:
                eps_tensor = elastic_cache.get("eps_tensor")
                if eps_tensor is None:
                    eps_tensor = tf.stack(
                        [
                            eps_vec[:, 0],
                            eps_vec[:, 1],
                            eps_vec[:, 2],
                            0.5 * eps_vec[:, 3],
                            0.5 * eps_vec[:, 4],
                            0.5 * eps_vec[:, 5],
                        ],
                        axis=1,
                    )
                else:
                    eps_tensor = tf.cast(eps_tensor, dtype)
                tr_eps = eps_tensor[:, 0] + eps_tensor[:, 1] + eps_tensor[:, 2]
                eye_vec = tf.constant([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=dtype)
                sigma_phys = lam[:, None] * tr_eps[:, None] * eye_vec + 2.0 * mu[:, None] * eps_tensor

            # ensure 6 components
            sigma_phys = sigma_phys[:, :6]
            # align to stress head order: xx, yy, zz, xy, yz, xz
            sigma_phys = tf.stack(
                [
                    sigma_phys[:, 0],
                    sigma_phys[:, 1],
                    sigma_phys[:, 2],
                    sigma_phys[:, 5],
                    sigma_phys[:, 3],
                    sigma_phys[:, 4],
                ],
                axis=1,
            )

            node_ids = tf.reshape(dof_idx // 3, (-1,))  # (M*4,)
            unique_nodes, rev = tf.unique(node_ids)
            X_nodes = tf.cast(tf.gather(self.elasticity.X_nodes_tf, unique_nodes), dtype)
            _, sigma_pred_nodes = stress_fn(X_nodes, params)
            sigma_pred_nodes = tf.cast(sigma_pred_nodes, dtype)

            #                           ?rev                          4                     ?            sigma_nodes_full = tf.gather(sigma_pred_nodes, rev)
            sigma_cells = tf.reshape(sigma_nodes_full, (tf.shape(dof_idx)[0], 4, -1))
            sigma_cells = tf.reduce_mean(sigma_cells, axis=1)
            sigma_cells = sigma_cells[:, : tf.shape(sigma_phys)[1]]
            #                                                         ?                    ?            sigma_ref = tf.cast(getattr(self.cfg, "sigma_ref", 1.0), dtype)
            sigma_ref = tf.maximum(sigma_ref, tf.cast(1e-12, dtype))

            if use_sigma:
                diff = sigma_cells - sigma_phys
                diff_n = diff / sigma_ref
                loss_sigma = tf.reduce_mean(diff_n * diff_n)
                parts["E_sigma"] = loss_sigma * tf.cast(
                    getattr(self.elasticity.cfg, "stress_loss_weight", 1.0), dtype
                )
                stats["stress_rms"] = tf.sqrt(tf.reduce_mean(sigma_cells * sigma_cells) + 1e-20)

                def _von_mises(sig: tf.Tensor) -> tf.Tensor:
                    sxx, syy, szz, sxy, syz, sxz = tf.unstack(sig, axis=1)
                    return tf.sqrt(
                        0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
                        + 3.0 * (sxy * sxy + syz * syz + sxz * sxz)
                    )

                vm_pred = _von_mises(sigma_cells)
                vm_phys = _von_mises(sigma_phys)
                stats["stress_vm_pred_max"] = tf.reduce_max(vm_pred)
                stats["stress_vm_pred_mean"] = tf.reduce_mean(vm_pred)
                stats["stress_vm_phys_max"] = tf.reduce_max(vm_phys)
                stats["stress_vm_phys_mean"] = tf.reduce_mean(vm_phys)

            if use_eq:
                sample_idx = elastic_cache.get("sample_idx")
                if sample_idx is not None:
                    sample_idx = tf.cast(sample_idx, tf.int32)
                    B_sel = tf.gather(self.elasticity.B_tf, sample_idx)
                    w_sel = tf.gather(self.elasticity.w_tf, sample_idx)
                else:
                    B_sel = tf.cast(self.elasticity.B_tf, dtype)
                    w_sel = tf.cast(self.elasticity.w_tf, dtype)

                # reorder to match B's shear ordering: (xx,yy,zz,yz,xz,xy)
                sigma_for_B = tf.stack(
                    [
                        sigma_cells[:, 0],
                        sigma_cells[:, 1],
                        sigma_cells[:, 2],
                        sigma_cells[:, 4],
                        sigma_cells[:, 5],
                        sigma_cells[:, 3],
                    ],
                    axis=1,
                )
                sigma_for_B = sigma_for_B / sigma_ref

                # element-wise internal force residual (weak form)
                f_int = tf.einsum("mij,mi->mj", B_sel, sigma_for_B)
                res = tf.reduce_sum(tf.square(f_int), axis=1)
                if w_sel is not None:
                    w_sel = tf.cast(w_sel, dtype)
                    res = res * w_sel
                    denom = tf.reduce_sum(w_sel) + tf.cast(1e-12, dtype)
                else:
                    denom = tf.cast(tf.shape(res)[0], dtype) + tf.cast(1e-12, dtype)
                parts["E_eq"] = tf.reduce_sum(res) / denom
                stats["eq_rms"] = tf.sqrt(tf.reduce_mean(res) + 1e-20)

        return parts, stats

    def _compute_parts_residual(self, u_fn, params, tape=None, stress_fn=None):
        """Evaluate residual-only components for a given parameter dictionary."""
        dtype = self.dtype
        zero = tf.cast(0.0, dtype)
        parts: Dict[str, tf.Tensor] = {
            "E_int": zero,
            "E_cn": zero,
            "E_ct": zero,
            "E_fb": zero,
            "E_region": zero,
            "E_tie": zero,
            "E_bc": zero,
            "W_pre": zero,
            "E_tight": zero,
            "E_sigma": zero,
            "E_eq": zero,
            "E_reg": zero,
        }
        stats: Dict[str, tf.Tensor] = {}

        u_nodes = None
        elastic_cache = None
        if self.elasticity is not None:
            u_nodes = self.elasticity._eval_u_on_nodes(u_fn, params)
            E_int_res = self.elasticity.energy(
                u_fn,
                params,
                tape=tape,
                return_cache=True,
                u_nodes=u_nodes,
            )
            _, estates, elastic_cache = E_int_res  # type: ignore[misc]
            stats.update({f"el_{k}": v for k, v in estates.items()})

        if self.contact is not None:
            _, cparts, stats_cn, stats_ct = self.contact.residual(u_fn, params, u_nodes=u_nodes)
            if "E_cn" in cparts:
                parts["E_cn"] = tf.cast(cparts["E_cn"], dtype)
            elif "E_n" in cparts:
                parts["E_cn"] = tf.cast(cparts["E_n"], dtype)
            if "E_ct" in cparts:
                parts["E_ct"] = tf.cast(cparts["E_ct"], dtype)
            elif "E_t" in cparts:
                parts["E_ct"] = tf.cast(cparts["E_t"], dtype)
            stats.update(stats_cn)
            stats.update(stats_ct)
            if "R_fric_comp" in stats_ct:
                parts["R_fric_comp"] = tf.cast(stats_ct["R_fric_comp"], dtype)
            if "R_contact_comp" in stats_cn:
                parts["R_contact_comp"] = tf.cast(stats_cn["R_contact_comp"], dtype)
            self._add_contact_aux_losses(
                u_fn, params, parts, stats, u_nodes=u_nodes
            )

        if self.ties:
            tie_terms = []
            for tie in self.ties:
                lt, tstat = tie.residual(u_fn, params)
                tie_terms.append(tf.cast(lt, dtype))
                if tstat:
                    for k, v in tstat.items():
                        stats[f"tie_{k}"] = v
            parts["E_tie"] = tf.add_n(tie_terms)

        if self.bcs:
            bc_terms = []
            for i, b in enumerate(self.bcs):
                if hasattr(b, "residual"):
                    L_bc_i, si = b.residual(u_fn, params)
                elif hasattr(b, "energy"):
                    L_bc_i, si = b.energy(u_fn, params)
                else:
                    continue
                bc_terms.append(tf.cast(L_bc_i, dtype))
                stats.update({f"bc{i+1}_{k}": v for k, v in si.items()})
            if bc_terms:
                parts["E_bc"] = tf.add_n(bc_terms)

        if self.preload is not None:
            L_pre, pstats = self.preload.residual(
                u_fn, params, u_nodes=u_nodes, stress_fn=stress_fn
            )
            parts["W_pre"] = tf.cast(L_pre, dtype)
            stats.update(pstats)
            stats.update({f"pre_{k}": v for k, v in pstats.items()})

        if self.tightening is not None:
            E_tight, tstats = self.tightening.residual(u_fn, params, u_nodes=u_nodes)
            parts["E_tight"] = tf.cast(E_tight, dtype)
            stats.update(tstats)

        w_sigma = float(getattr(self.cfg, "w_sigma", 0.0))
        w_eq = float(getattr(self.cfg, "w_eq", 0.0))
        w_reg = float(getattr(self.cfg, "w_reg", 0.0))
        use_stress = stress_fn is not None and elastic_cache is not None
        use_sigma = use_stress and w_sigma > 1e-15 and getattr(self.elasticity.cfg, "stress_loss_weight", 0.0) > 0.0
        use_eq = elastic_cache is not None and w_eq > 1e-15
        use_reg = elastic_cache is not None and w_reg > 1e-15

        if use_sigma or use_eq or use_reg:
            eps_vec = tf.cast(elastic_cache["eps_vec"], dtype)
            lam = tf.cast(elastic_cache["lam"], dtype)
            mu = tf.cast(elastic_cache["mu"], dtype)
            dof_idx = tf.cast(elastic_cache["dof_idx"], tf.int32)

            sigma_phys_raw = elastic_cache.get("sigma_phys")
            if sigma_phys_raw is not None:
                sigma_phys_raw = tf.cast(sigma_phys_raw, dtype)
            else:
                eps_tensor = elastic_cache.get("eps_tensor")
                if eps_tensor is None:
                    eps_tensor = tf.stack(
                        [
                            eps_vec[:, 0],
                            eps_vec[:, 1],
                            eps_vec[:, 2],
                            0.5 * eps_vec[:, 3],
                            0.5 * eps_vec[:, 4],
                            0.5 * eps_vec[:, 5],
                        ],
                        axis=1,
                    )
                else:
                    eps_tensor = tf.cast(eps_tensor, dtype)
                tr_eps = eps_tensor[:, 0] + eps_tensor[:, 1] + eps_tensor[:, 2]
                eye_vec = tf.constant([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=dtype)
                sigma_phys_raw = lam[:, None] * tr_eps[:, None] * eye_vec + 2.0 * mu[:, None] * eps_tensor

            sigma_phys_raw = sigma_phys_raw[:, :6]
            sigma_phys_head = tf.stack(
                [
                    sigma_phys_raw[:, 0],
                    sigma_phys_raw[:, 1],
                    sigma_phys_raw[:, 2],
                    sigma_phys_raw[:, 5],
                    sigma_phys_raw[:, 3],
                    sigma_phys_raw[:, 4],
                ],
                axis=1,
            )

            sigma_cells = None
            if use_stress:
                node_ids = tf.reshape(dof_idx // 3, (-1,))
                unique_nodes, rev = tf.unique(node_ids)
                X_nodes = tf.cast(tf.gather(self.elasticity.X_nodes_tf, unique_nodes), dtype)
                _, sigma_pred_nodes = stress_fn(X_nodes, params)
                sigma_pred_nodes = tf.cast(sigma_pred_nodes, dtype)

                sigma_nodes_full = tf.gather(sigma_pred_nodes, rev)
                sigma_cells = tf.reshape(sigma_nodes_full, (tf.shape(dof_idx)[0], 4, -1))
                sigma_cells = tf.reduce_mean(sigma_cells, axis=1)
                sigma_cells = sigma_cells[:, : tf.shape(sigma_phys_head)[1]]

            sigma_ref = tf.cast(getattr(self.cfg, "sigma_ref", 1.0), dtype)
            sigma_ref = tf.maximum(sigma_ref, tf.cast(1e-12, dtype))

            if use_sigma and sigma_cells is not None:
                diff = sigma_cells - sigma_phys_head
                diff_n = diff / sigma_ref
                loss_sigma = tf.reduce_mean(diff_n * diff_n)
                parts["E_sigma"] = loss_sigma * tf.cast(
                    getattr(self.elasticity.cfg, "stress_loss_weight", 1.0), dtype
                )
                stats["stress_rms"] = tf.sqrt(tf.reduce_mean(sigma_cells * sigma_cells) + 1e-20)

            if use_eq:
                sample_idx = elastic_cache.get("sample_idx")
                if sample_idx is not None:
                    sample_idx = tf.cast(sample_idx, tf.int32)
                    B_sel = tf.gather(self.elasticity.B_tf, sample_idx)
                    w_sel = tf.gather(self.elasticity.w_tf, sample_idx)
                else:
                    B_sel = tf.cast(self.elasticity.B_tf, dtype)
                    w_sel = tf.cast(self.elasticity.w_tf, dtype)

                if sigma_cells is not None:
                    sigma_for_B = tf.stack(
                        [
                            sigma_cells[:, 0],
                            sigma_cells[:, 1],
                            sigma_cells[:, 2],
                            sigma_cells[:, 4],
                            sigma_cells[:, 5],
                            sigma_cells[:, 3],
                        ],
                        axis=1,
                    )
                else:
                    sigma_for_B = tf.cast(sigma_phys_raw, dtype)

                sigma_for_B = sigma_for_B / sigma_ref
                f_int = tf.einsum("mij,mi->mj", B_sel, sigma_for_B)
                res = tf.reduce_sum(tf.square(f_int), axis=1)
                if w_sel is not None:
                    w_sel = tf.cast(w_sel, dtype)
                    res = res * w_sel
                    denom = tf.reduce_sum(w_sel) + tf.cast(1e-12, dtype)
                else:
                    denom = tf.cast(tf.shape(res)[0], dtype) + tf.cast(1e-12, dtype)
                parts["E_eq"] = tf.reduce_sum(res) / denom
                stats["eq_rms"] = tf.sqrt(tf.reduce_mean(res) + 1e-20)

            if use_reg:
                w_sel = elastic_cache.get("w_sel")
                res = tf.reduce_sum(eps_vec * eps_vec, axis=1)
                if w_sel is not None:
                    w_sel = tf.cast(w_sel, dtype)
                    res = res * w_sel
                    denom = tf.reduce_sum(w_sel) + tf.cast(1e-12, dtype)
                else:
                    denom = tf.cast(tf.shape(res)[0], dtype) + tf.cast(1e-12, dtype)
                parts["E_reg"] = tf.reduce_sum(res) / denom

        return parts, stats
    def _combine_parts(self, parts: Dict[str, tf.Tensor]) -> tf.Tensor:
        def _safe_part(name: str) -> tf.Tensor:
            raw = parts.get(name, tf.cast(0.0, self.dtype))
            try:
                val = tf.cast(raw, self.dtype)
            except Exception:
                val = tf.cast(0.0, self.dtype)
            return tf.where(tf.math.is_finite(val), val, tf.cast(0.0, self.dtype))

        pre_sign = tf.cast(-1.0 if self._loss_mode() == "energy" else 1.0, self.dtype)
        return (
            self.w_int * _safe_part("E_int")
            + self.w_cn * _safe_part("E_cn")
            + self.w_ct * _safe_part("E_ct")
            + self.w_fb * _safe_part("E_fb")
            + self.w_region * _safe_part("E_region")
            + self.w_tie * _safe_part("E_tie")
            + self.w_bc * _safe_part("E_bc")
            + pre_sign * self.w_pre * _safe_part("W_pre")
            + self.w_tight * _safe_part("E_tight")
            + self.w_sigma * _safe_part("E_sigma")
            + self.w_eq * _safe_part("E_eq")
            + self.w_reg * _safe_part("E_reg")
        )

    def _combine_parts_without_preload(self, parts: Dict[str, tf.Tensor]) -> tf.Tensor:
        """   _combine_parts                                         """

        def _safe_part(name: str) -> tf.Tensor:
            raw = parts.get(name, tf.cast(0.0, self.dtype))
            try:
                val = tf.cast(raw, self.dtype)
            except Exception:
                val = tf.cast(0.0, self.dtype)
            return tf.where(tf.math.is_finite(val), val, tf.cast(0.0, self.dtype))

        return (
            self.w_int * _safe_part("E_int")
            + self.w_cn * _safe_part("E_cn")
            + self.w_ct * _safe_part("E_ct")
            + self.w_fb * _safe_part("E_fb")
            + self.w_region * _safe_part("E_region")
            + self.w_tie * _safe_part("E_tie")
            + self.w_bc * _safe_part("E_bc")
            + self.w_tight * _safe_part("E_tight")
            + self.w_sigma * _safe_part("E_sigma")
            + self.w_eq * _safe_part("E_eq")
            + self.w_reg * _safe_part("E_reg")
        )

    def _energy_staged(self, u_fn, stages, root_params, tape=None, stress_fn=None):
        """Accumulate energy across staged preload applications.

                                                                                                 ?          _step,i = (E_int + E_cn + E_ct + E_tie)_i -   W_pre,i
                                  ?                                                                          ?                                                   ALM                                    ?        """
        dtype = self.dtype
        keys = [
            "E_int",
            "E_cn",
            "E_ct",
            "E_fb",
            "E_region",
            "E_tie",
            "E_bc",
            "W_pre",
            "E_tight",
            "E_sigma",
            "E_eq",
        ]
        totals: Dict[str, tf.Tensor] = {k: tf.cast(0.0, dtype) for k in keys}
        stats_all: Dict[str, tf.Tensor] = {}
        path_penalty_total = tf.cast(0.0, dtype)
        fric_path_penalty_total = tf.cast(0.0, dtype)
        Pi_accum = tf.cast(0.0, dtype)

        stage_mode = str(getattr(self.cfg, "preload_stage_mode", "cumulative_force") or "cumulative_force")
        stage_mode = stage_mode.strip().lower().replace("-", "_")
        force_then_lock = stage_mode == "force_then_lock"

        if isinstance(stages, dict):
            stage_tensor_P = stages.get("P")
            stage_tensor_feat = stages.get("P_hat")
            stage_tensor_rank = stages.get("stage_rank")
            stage_tensor_mask = stages.get("stage_mask")
            stage_tensor_last = stages.get("stage_last")
            if stage_tensor_P is None or stage_tensor_feat is None:
                stage_seq: List[Dict[str, tf.Tensor]] = []
            else:
                stacked_rank = None
                if stage_tensor_rank is not None:
                    stacked_rank = tf.convert_to_tensor(stage_tensor_rank)
                stacked_mask = None
                if stage_tensor_mask is not None:
                    stacked_mask = tf.convert_to_tensor(stage_tensor_mask)
                stacked_last = None
                if stage_tensor_last is not None:
                    stacked_last = tf.convert_to_tensor(stage_tensor_last)
                stage_seq = []
                for idx, (p, z) in enumerate(
                    zip(tf.unstack(stage_tensor_P, axis=0), tf.unstack(stage_tensor_feat, axis=0))
                ):
                    entry = {"P": p, "P_hat": z}
                    if stacked_rank is not None:
                        if stacked_rank.shape.rank == 2:
                            entry["stage_rank"] = stacked_rank[idx]
                        else:
                            entry["stage_rank"] = stacked_rank
                    if stacked_mask is not None:
                        entry["stage_mask"] = stacked_mask[idx]
                    if stacked_last is not None:
                        entry["stage_last"] = stacked_last[idx]
                    stage_seq.append(entry)
        else:
            stage_seq = []
            for item in stages:
                if isinstance(item, dict):
                    stage_seq.append(item)
                else:
                    p_val, z_val = item
                    stage_seq.append({"P": p_val, "P_hat": z_val})

        if not stage_seq:
            return self._combine_parts(totals), totals, stats_all

        prev_bolt_deltas: Optional[tf.Tensor] = None
        prev_P: Optional[tf.Tensor] = None
        prev_slip: Optional[tf.Tensor] = None
        prev_W_pre = tf.cast(0.0, dtype)
        last_preload_entry: Optional[dict] = None

        stage_count = len(stage_seq)

        for idx, stage_params in enumerate(stage_seq):
            #                                                                   
            stage_idx = tf.cast(idx, tf.int32)
            stage_frac = tf.cast(
                0.0 if stage_count <= 1 else idx / max(stage_count - 1, 1), dtype
            )
            stage_params = dict(stage_params)
            stage_params.setdefault("stage_index", stage_idx)
            stage_params.setdefault("stage_fraction", stage_frac)
            if isinstance(root_params, dict) and "train_progress" in root_params:
                stage_params.setdefault("train_progress", root_params["train_progress"])

            # "force_then_lock": only the currently tightened bolt is force-controlled.
            # Keep the cumulative target vector for path penalties/diagnostics.
            if force_then_lock:
                stage_last = stage_params.get("stage_last")
                if stage_last is not None and "P" in stage_params:
                    P_cumulative = tf.convert_to_tensor(stage_params["P"], dtype=tf.float32)
                    stage_params["P_cumulative"] = P_cumulative
                    stage_params["P"] = P_cumulative * tf.cast(stage_last, P_cumulative.dtype)

            stage_parts, stage_stats = self._compute_parts(
                u_fn, stage_params, tape, stress_fn=stress_fn
            )
            stage_parts = dict(stage_parts)
            for k, v in stage_stats.items():
                stats_all[f"s{idx+1}_{k}"] = v

            bolt_deltas = None
            pre_entry = stage_stats.get("preload")
            if pre_entry is None:
                pre_entry = stage_stats.get("preload_stats")
            if pre_entry is None:
                pre_entry = stage_stats.get("pre_preload")
            if isinstance(pre_entry, dict):
                bd = pre_entry.get("bolt_deltas")
                if bd is None:
                    bd = pre_entry.get("bolt_delta")
                if bd is not None:
                    bolt_deltas = tf.cast(bd, dtype)
                    last_preload_entry = pre_entry

            # No lock penalty: earlier bolts are free to relax.

            W_stage = tf.cast(stage_parts.get("W_pre", tf.cast(0.0, dtype)), dtype)
            if force_then_lock and "P_cumulative" in stage_params:
                stats_all[f"s{idx+1}_W_pre_stage"] = W_stage

            for key in keys:
                if key == "W_pre" and force_then_lock and "P_cumulative" in stage_params:
                    cur = prev_W_pre + W_stage
                else:
                    cur = tf.cast(stage_parts.get(key, tf.cast(0.0, dtype)), dtype)
                # staged objective uses   W_pre, so for adaptive combine_loss we expose the final W_pre;
                # other energy parts remain stage-summed to match   _step accumulation.
                if key == "W_pre":
                    totals[key] = cur
                else:
                    totals[key] = totals[key] + cur  #                                       ?
                stats_all[f"s{idx+1}_{key}"] = cur
                stats_all[f"s{idx+1}_cum{key}"] = totals[key]

            P_vec = tf.cast(
                tf.convert_to_tensor(stage_params.get("P_cumulative", stage_params.get("P", []))),
                dtype,
            )
            slip_t = None
            if self.contact is not None and hasattr(self.contact, "last_friction_slip"):
                slip_t = self.contact.last_friction_slip()

            w_path = tf.cast(getattr(self.cfg, "path_penalty_weight", 1.0), dtype)
            w_fric_path = tf.cast(getattr(self.cfg, "fric_path_penalty_weight", 1.0), dtype)

            stage_path_penalty = tf.cast(0.0, dtype)
            stage_path = tf.cast(0.0, dtype)
            stage_fric_path = tf.cast(0.0, dtype)
            if idx > 0:
                load_jump = tf.reduce_sum(tf.abs(P_vec - prev_P)) if prev_P is not None else tf.cast(0.0, dtype)

                if bolt_deltas is not None and prev_bolt_deltas is not None:
                    disp_jump = tf.reduce_sum(tf.abs(bolt_deltas - prev_bolt_deltas))
                    stage_path = disp_jump * load_jump
                    path_penalty_total = path_penalty_total + stage_path
                    stats_all[f"s{idx+1}_path_penalty"] = stage_path
                    stats_all[f"s{idx+1}_path_penalty_w"] = w_path

                if slip_t is not None and prev_slip is not None:
                    slip_jump = tf.reduce_sum(tf.abs(slip_t - prev_slip))
                    stage_fric_path = slip_jump * load_jump
                    fric_path_penalty_total = fric_path_penalty_total + stage_fric_path
                    stats_all[f"s{idx+1}_fric_path_penalty"] = stage_fric_path
                    stats_all[f"s{idx+1}_fric_path_penalty_w"] = w_fric_path
            stage_path_penalty = stage_path_penalty + w_path * stage_path + w_fric_path * stage_fric_path

            W_cur = tf.cast(stage_parts.get("W_pre", tf.cast(0.0, dtype)), dtype)
            if force_then_lock and "P_cumulative" in stage_params:
                W_cur = prev_W_pre + W_stage
            delta_W = W_cur - prev_W_pre
            stage_mech = self._combine_parts_without_preload(stage_parts)

            stage_pi_step = stage_mech - self.w_pre * delta_W + stage_path_penalty
            stats_all[f"s{idx+1}_Pi_step"] = stage_pi_step
            stats_all[f"s{idx+1}_delta_W_pre"] = delta_W
            stats_all[f"s{idx+1}_Pi_mech"] = stage_mech

            Pi_accum = Pi_accum + stage_pi_step

            if bolt_deltas is not None:
                prev_bolt_deltas = bolt_deltas
            if tf.size(P_vec) > 0:
                prev_P = P_vec
            if slip_t is not None:
                prev_slip = slip_t
            prev_W_pre = W_cur
            if self.contact is not None:
                try:
                    stage_params_detached = {
                        k: tf.stop_gradient(v) if isinstance(v, tf.Tensor) else v for k, v in stage_params.items()
                    }
                    self.contact.update_multipliers(u_fn, stage_params_detached)
                except Exception:
                    pass
            if self.bcs:
                for bc in self.bcs:
                    try:
                        bc.update_multipliers(u_fn, stage_params)
                    except Exception:
                        pass

        if isinstance(root_params, dict):
            if "stage_order" in root_params:
                stats_all["stage_order"] = root_params["stage_order"]
            if "stage_rank" in root_params:
                stats_all["stage_rank"] = root_params["stage_rank"]
            if "stage_count" in root_params:
                stats_all["stage_count"] = root_params["stage_count"]

        stats_all["path_penalty_total"] = path_penalty_total
        stats_all["fric_path_penalty_total"] = fric_path_penalty_total
        totals["path_penalty_total"] = path_penalty_total
        totals["fric_path_penalty_total"] = fric_path_penalty_total
        if isinstance(last_preload_entry, dict):
            stats_all["preload"] = last_preload_entry

        Pi = Pi_accum
        return Pi, totals, stats_all

    def _residual_staged(self, u_fn, stages, root_params, tape=None, stress_fn=None):
        """Accumulate residual-only loss across staged preload applications."""
        dtype = self.dtype
        keys = [
            "E_int",
            "E_cn",
            "E_ct",
            "E_fb",
            "E_region",
            "E_tie",
            "E_bc",
            "W_pre",
            "E_tight",
            "E_sigma",
            "E_eq",
            "E_reg",
        ]
        totals: Dict[str, tf.Tensor] = {k: tf.cast(0.0, dtype) for k in keys}
        stats_all: Dict[str, tf.Tensor] = {}
        path_penalty_total = tf.cast(0.0, dtype)
        fric_path_penalty_total = tf.cast(0.0, dtype)
        Pi_accum = tf.cast(0.0, dtype)

        stage_mode = str(getattr(self.cfg, "preload_stage_mode", "cumulative_force") or "cumulative_force")
        stage_mode = stage_mode.strip().lower().replace("-", "_")
        force_then_lock = stage_mode == "force_then_lock"

        if isinstance(stages, dict):
            stage_tensor_P = stages.get("P")
            stage_tensor_feat = stages.get("P_hat")
            stage_tensor_rank = stages.get("stage_rank")
            stage_tensor_mask = stages.get("stage_mask")
            stage_tensor_last = stages.get("stage_last")
            if stage_tensor_P is None or stage_tensor_feat is None:
                stage_seq: List[Dict[str, tf.Tensor]] = []
            else:
                stacked_rank = None
                if stage_tensor_rank is not None:
                    stacked_rank = tf.convert_to_tensor(stage_tensor_rank)
                stacked_mask = None
                if stage_tensor_mask is not None:
                    stacked_mask = tf.convert_to_tensor(stage_tensor_mask)
                stacked_last = None
                if stage_tensor_last is not None:
                    stacked_last = tf.convert_to_tensor(stage_tensor_last)
                stage_seq = []
                for idx, (p, z) in enumerate(
                    zip(tf.unstack(stage_tensor_P, axis=0), tf.unstack(stage_tensor_feat, axis=0))
                ):
                    entry = {"P": p, "P_hat": z}
                    if stacked_rank is not None:
                        if stacked_rank.shape.rank == 2:
                            entry["stage_rank"] = stacked_rank[idx]
                        else:
                            entry["stage_rank"] = stacked_rank
                    if stacked_mask is not None:
                        entry["stage_mask"] = stacked_mask[idx]
                    if stacked_last is not None:
                        entry["stage_last"] = stacked_last[idx]
                    stage_seq.append(entry)
        else:
            stage_seq = []
            for item in stages:
                if isinstance(item, dict):
                    stage_seq.append(item)
                else:
                    p_val, z_val = item
                    stage_seq.append({"P": p_val, "P_hat": z_val})

        if not stage_seq:
            return self._combine_parts(totals), totals, stats_all

        prev_bolt_deltas: Optional[tf.Tensor] = None
        prev_P: Optional[tf.Tensor] = None
        prev_slip: Optional[tf.Tensor] = None
        prev_W_pre = tf.cast(0.0, dtype)
        last_preload_entry: Optional[dict] = None

        stage_count = len(stage_seq)

        for idx, stage_params in enumerate(stage_seq):
            stage_idx = tf.cast(idx, tf.int32)
            stage_frac = tf.cast(
                0.0 if stage_count <= 1 else idx / max(stage_count - 1, 1), dtype
            )
            stage_params = dict(stage_params)
            stage_params.setdefault("stage_index", stage_idx)
            stage_params.setdefault("stage_fraction", stage_frac)
            if isinstance(root_params, dict) and "train_progress" in root_params:
                stage_params.setdefault("train_progress", root_params["train_progress"])

            if force_then_lock:
                stage_last = stage_params.get("stage_last")
                if stage_last is not None and "P" in stage_params:
                    P_cumulative = tf.convert_to_tensor(stage_params["P"], dtype=tf.float32)
                    stage_params["P_cumulative"] = P_cumulative
                    stage_params["P"] = P_cumulative * tf.cast(stage_last, P_cumulative.dtype)

            stage_parts, stage_stats = self._compute_parts_residual(
                u_fn, stage_params, tape, stress_fn=stress_fn
            )
            stage_parts = dict(stage_parts)
            for k, v in stage_stats.items():
                stats_all[f"s{idx+1}_{k}"] = v

            bolt_deltas = None
            pre_entry = stage_stats.get("preload")
            if pre_entry is None:
                pre_entry = stage_stats.get("preload_stats")
            if pre_entry is None:
                pre_entry = stage_stats.get("pre_preload")
            if isinstance(pre_entry, dict):
                bd = pre_entry.get("bolt_deltas")
                if bd is None:
                    bd = pre_entry.get("bolt_delta")
                if bd is not None:
                    bolt_deltas = tf.cast(bd, dtype)
                    last_preload_entry = pre_entry

            # No lock penalty: earlier bolts are free to relax.

            W_stage = tf.cast(stage_parts.get("W_pre", tf.cast(0.0, dtype)), dtype)
            if force_then_lock and "P_cumulative" in stage_params:
                stats_all[f"s{idx+1}_W_pre_stage"] = W_stage

            if force_then_lock and "P_cumulative" in stage_params:
                W_cur = prev_W_pre + W_stage
            else:
                W_cur = W_stage
            delta_W = W_cur - prev_W_pre

            for key in keys:
                if key == "W_pre":
                    cur = W_cur
                    totals[key] = cur
                else:
                    cur = tf.cast(stage_parts.get(key, tf.cast(0.0, dtype)), dtype)
                    totals[key] = totals[key] + cur
                stats_all[f"s{idx+1}_{key}"] = cur
                stats_all[f"s{idx+1}_cum{key}"] = totals[key]

            P_vec = tf.cast(
                tf.convert_to_tensor(stage_params.get("P_cumulative", stage_params.get("P", []))),
                dtype,
            )
            slip_t = None
            if self.contact is not None and hasattr(self.contact, "last_friction_slip"):
                slip_t = self.contact.last_friction_slip()

            w_path = tf.cast(getattr(self.cfg, "path_penalty_weight", 1.0), dtype)
            w_fric_path = tf.cast(getattr(self.cfg, "fric_path_penalty_weight", 1.0), dtype)

            stage_path = tf.cast(0.0, dtype)
            stage_fric_path = tf.cast(0.0, dtype)
            if idx > 0:
                load_jump = tf.reduce_sum(tf.abs(P_vec - prev_P)) if prev_P is not None else tf.cast(0.0, dtype)
                if bolt_deltas is not None and prev_bolt_deltas is not None:
                    disp_jump = tf.reduce_sum(tf.abs(bolt_deltas - prev_bolt_deltas))
                    stage_path = disp_jump * load_jump
                    path_penalty_total = path_penalty_total + stage_path
                    stats_all[f"s{idx+1}_path_penalty"] = stage_path
                    stats_all[f"s{idx+1}_path_penalty_w"] = w_path
                if slip_t is not None and prev_slip is not None:
                    slip_jump = tf.reduce_sum(tf.abs(slip_t - prev_slip))
                    stage_fric_path = slip_jump * load_jump
                    fric_path_penalty_total = fric_path_penalty_total + stage_fric_path
                    stats_all[f"s{idx+1}_fric_path_penalty"] = stage_fric_path
                    stats_all[f"s{idx+1}_fric_path_penalty_w"] = w_fric_path

            stage_path_penalty = w_path * stage_path + w_fric_path * stage_fric_path

            stage_parts_delta = dict(stage_parts)
            stage_parts_delta["W_pre"] = delta_W
            stage_parts_mech = dict(stage_parts)
            stage_parts_mech["W_pre"] = tf.cast(0.0, dtype)
            stage_mech = self._combine_parts(stage_parts_mech)
            stage_pi_step = self._combine_parts(stage_parts_delta) + stage_path_penalty
            stats_all[f"s{idx+1}_Pi_step"] = stage_pi_step
            stats_all[f"s{idx+1}_delta_W_pre"] = delta_W
            stats_all[f"s{idx+1}_Pi_mech"] = stage_mech

            Pi_accum = Pi_accum + stage_pi_step

            if bolt_deltas is not None:
                prev_bolt_deltas = bolt_deltas
            prev_P = P_vec
            if slip_t is not None:
                prev_slip = slip_t
            prev_W_pre = W_cur

        stats_all["path_penalty_total"] = path_penalty_total
        stats_all["fric_path_penalty_total"] = fric_path_penalty_total
        totals["path_penalty_total"] = path_penalty_total
        totals["fric_path_penalty_total"] = fric_path_penalty_total
        if isinstance(last_preload_entry, dict):
            stats_all["preload"] = last_preload_entry

        Pi = Pi_accum
        return Pi, totals, stats_all

    # ---------- outer updates ----------
    def update_multipliers(self, u_fn, params=None):
        """
        Run ALM outer-loop updates for contact (and anything else in future).
        Call this every cfg.update_every_steps steps in your training loop.
        """
        target_params = params
        staged_updates: List[Dict[str, tf.Tensor]] = []
        if isinstance(params, dict) and params.get("stages"):
            stages = params["stages"]
            if isinstance(stages, dict):
                stage_tensor_P = stages.get("P")
                stage_tensor_feat = stages.get("P_hat")
                stage_tensor_rank = stages.get("stage_rank")
                if stage_tensor_P is not None and stage_tensor_feat is not None:
                    for idx, (p, z) in enumerate(
                        zip(tf.unstack(stage_tensor_P, axis=0), tf.unstack(stage_tensor_feat, axis=0))
                    ):
                        entry: Dict[str, tf.Tensor] = {"P": p, "P_hat": z}
                        if stage_tensor_rank is not None:
                            if stage_tensor_rank.shape.rank == 2:
                                entry["stage_rank"] = stage_tensor_rank[idx]
                            else:
                                entry["stage_rank"] = stage_tensor_rank
                        staged_updates.append(entry)
                        target_params = entry
            elif isinstance(stages, (list, tuple)) and stages:
                for stage in stages:
                    if isinstance(stage, dict):
                        staged_updates.append(stage)
                        target_params = stage
                    else:
                        p_val, z_val = stage
                        entry = {"P": p_val, "P_hat": z_val}
                        staged_updates.append(entry)
                        target_params = entry

        if self.contact is not None:
            if staged_updates:
                for st_params in staged_updates:
                    u_nodes = None
                    if self.elasticity is not None:
                        u_nodes = self.elasticity._eval_u_on_nodes(u_fn, st_params)
                    self.contact.update_multipliers(u_fn, st_params, u_nodes=u_nodes)
            else:
                u_nodes = None
                if self.elasticity is not None:
                    u_nodes = self.elasticity._eval_u_on_nodes(u_fn, target_params)
                self.contact.update_multipliers(u_fn, target_params, u_nodes=u_nodes)
        if self.bcs:
            if staged_updates:
                for st_params in staged_updates:
                    for bc in self.bcs:
                        updater = getattr(bc, "update_multipliers", None)
                        if callable(updater):
                            updater(u_fn, st_params)
            else:
                for bc in self.bcs:
                    updater = getattr(bc, "update_multipliers", None)
                    if callable(updater):
                        updater(u_fn, target_params)

    # ---------- setters / schedules ----------

    def set_coeffs(
        self,
        w_int: Optional[float] = None,
        w_cn: Optional[float] = None,
        w_ct: Optional[float] = None,
        w_tie: Optional[float] = None,
        w_pre: Optional[float] = None,
    ):
        """Set any subset of coefficients on the fly (e.g., curriculum)."""
        if w_int is not None:
            self.w_int.assign(tf.cast(w_int, self.dtype))
        if w_cn is not None:
            self.w_cn.assign(tf.cast(w_cn, self.dtype))
        if w_ct is not None:
            self.w_ct.assign(tf.cast(w_ct, self.dtype))
        if w_tie is not None:
            self.w_tie.assign(tf.cast(w_tie, self.dtype))
        if w_pre is not None:
            self.w_pre.assign(tf.cast(w_pre, self.dtype))


# -----------------------------
# Minimal smoke test
# -----------------------------
if __name__ == "__main__":
    pass
