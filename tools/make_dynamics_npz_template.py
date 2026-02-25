#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a dynamics dataset template (.npz) from an INP/CDB mesh.

This script creates placeholder arrays for time-series structural dynamics:
    x0, node_id, u, v, a, f_ext, t, dt, fixed_mask, observed_mask, ...

Usage:
  python tools/make_dynamics_npz_template.py --mesh C:/codex/mir111.cdb
  python tools/make_dynamics_npz_template.py --mesh shuangfan.inp --timesteps 400 --dt 2e-4
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np


def _normalize_mesh_path(path_raw: str) -> str:
    p = str(path_raw or "").strip().strip('"').strip("'")
    if not p:
        return p
    if os.path.exists(p):
        return p

    if os.name != "nt":
        m = re.match(r"^([A-Za-z]):[\\/](.*)$", p)
        if m:
            drive = m.group(1).lower()
            rest = m.group(2).replace("\\", "/")
            return f"/mnt/{drive}/{rest}"
    else:
        m = re.match(r"^/mnt/([A-Za-z])/(.*)$", p)
        if m:
            drive = m.group(1).upper()
            rest = m.group(2).replace("/", "\\")
            return f"{drive}:\\{rest}"
    return p


def _load_assembly(mesh_path: Path) -> Any:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(repo_root / "src"))

    ext = mesh_path.suffix.lower()
    if ext == ".cdb":
        from inp_io.cdb_parser import load_cdb  # type: ignore

        return load_cdb(str(mesh_path))
    from inp_io.inp_parser import load_inp  # type: ignore

    return load_inp(str(mesh_path))


def _extract_nodes(asm: Any) -> Tuple[np.ndarray, np.ndarray]:
    nodes = getattr(asm, "nodes", {}) or {}
    if not nodes:
        raise ValueError("Assembly has no nodes.")
    node_ids = np.asarray(sorted(int(k) for k in nodes.keys()), dtype=np.int64)
    x0 = np.asarray([nodes[int(nid)] for nid in node_ids], dtype=np.float32)
    if x0.ndim != 2 or x0.shape[1] != 3:
        raise ValueError(f"Unexpected node coordinates shape: {x0.shape}")
    return node_ids, x0


def _extract_parts(asm: Any, node_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    parts = getattr(asm, "parts", {}) or {}
    part_names = sorted(str(k) for k in parts.keys())
    if not part_names:
        return np.asarray([], dtype="<U1"), np.full((node_ids.shape[0],), -1, dtype=np.int32)

    node_to_index = {int(nid): i for i, nid in enumerate(node_ids.tolist())}
    part_index = np.full((node_ids.shape[0],), -1, dtype=np.int32)

    for pi, pname in enumerate(part_names):
        p = parts.get(pname, None)
        if p is None:
            continue
        for nid in getattr(p, "node_ids", []) or []:
            j = node_to_index.get(int(nid), None)
            if j is None:
                continue
            if part_index[j] < 0:
                part_index[j] = int(pi)

    return np.asarray(part_names, dtype="<U128"), part_index


def _extract_edges(asm: Any, node_ids: np.ndarray) -> np.ndarray:
    """
    Build undirected mesh edges from element connectivity:
    connect each pair of consecutive nodes in each element loop.
    """
    node_to_index = {int(nid): i for i, nid in enumerate(node_ids.tolist())}
    elements = getattr(asm, "elements", {}) or {}
    edge_set = set()

    for _, conn in elements.items():
        conn_list: List[int] = [int(v) for v in conn if int(v) in node_to_index]
        if len(conn_list) < 2:
            continue
        # Local loop edges (much lighter than clique edges).
        for i in range(len(conn_list)):
            a = int(conn_list[i])
            b = int(conn_list[(i + 1) % len(conn_list)])
            if a == b:
                continue
            ia = node_to_index[a]
            ib = node_to_index[b]
            lo, hi = (ia, ib) if ia < ib else (ib, ia)
            edge_set.add((lo, hi))

    if not edge_set:
        return np.zeros((2, 0), dtype=np.int64)

    undirected = sorted(edge_set)
    src: List[int] = []
    dst: List[int] = []
    for a, b in undirected:
        src.append(a)
        dst.append(b)
        src.append(b)
        dst.append(a)
    return np.asarray([src, dst], dtype=np.int64)


def _build_template(
    asm: Any,
    timesteps: int,
    dt: float,
    include_edges: bool,
) -> Dict[str, np.ndarray]:
    node_ids, x0 = _extract_nodes(asm)
    n = int(node_ids.shape[0])
    t = np.arange(int(timesteps), dtype=np.float32) * np.float32(dt)

    u = np.zeros((timesteps, n, 3), dtype=np.float32)
    v = np.zeros((timesteps, n, 3), dtype=np.float32)
    a = np.zeros((timesteps, n, 3), dtype=np.float32)
    f_ext = np.zeros((timesteps, n, 3), dtype=np.float32)

    fixed_mask = np.zeros((n, 3), dtype=np.uint8)
    observed_mask = np.ones((timesteps, n, 3), dtype=np.uint8)

    part_names, node_part_index = _extract_parts(asm, node_ids)
    edge_index = _extract_edges(asm, node_ids) if include_edges else np.zeros((2, 0), dtype=np.int64)

    data: Dict[str, np.ndarray] = {
        "node_id": node_ids,
        "x0": x0,
        "t": t,
        "dt": np.asarray([dt], dtype=np.float32),
        "u": u,
        "v": v,
        "a": a,
        "f_ext": f_ext,
        "fixed_mask": fixed_mask,
        "observed_mask": observed_mask,
        "node_part_index": node_part_index,
        "part_names": part_names,
        "edge_index": edge_index,
    }
    return data


def _estimate_bytes(timesteps: int, n_nodes: int) -> int:
    # u/v/a/f_ext: 4 arrays, float32, shape=(T,N,3)
    return int(4 * timesteps * n_nodes * 3 * 4)


def _schema_for(data: Dict[str, np.ndarray]) -> Dict[str, Any]:
    fields: Dict[str, Any] = {}
    for k, v in data.items():
        fields[k] = {"shape": list(v.shape), "dtype": str(v.dtype)}
    fields["notes"] = {
        "u": "displacement time series, shape=(T,N,3)",
        "v": "velocity time series, shape=(T,N,3)",
        "a": "acceleration time series, shape=(T,N,3)",
        "f_ext": "external nodal force time series, shape=(T,N,3)",
        "fixed_mask": "1 means constrained dof, shape=(N,3)",
        "observed_mask": "1 means supervised entry, shape=(T,N,3)",
        "edge_index": "directed COO edges, shape=(2,E), may be empty",
    }
    return fields


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Create a dynamics .npz template from INP/CDB.")
    ap.add_argument("--mesh", required=True, help="Path to mesh file (.inp or .cdb)")
    ap.add_argument("--out", default="dynamics_template.npz", help="Output .npz path")
    ap.add_argument("--timesteps", type=int, default=8, help="Number of time steps T")
    ap.add_argument("--dt", type=float, default=1.0e-4, help="Time step size")
    ap.add_argument(
        "--max-gb",
        type=float,
        default=2.0,
        help="Abort if estimated u/v/a/f_ext memory exceeds this threshold (GB)",
    )
    ap.add_argument(
        "--allow-large",
        action="store_true",
        help="Allow generating template even if estimated memory exceeds --max-gb",
    )
    ap.add_argument(
        "--include-edges",
        action="store_true",
        help="Build a lightweight edge_index from element local loops",
    )
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    if args.timesteps <= 1:
        raise ValueError("--timesteps must be >= 2")
    if args.dt <= 0:
        raise ValueError("--dt must be > 0")

    mesh_path = Path(_normalize_mesh_path(args.mesh)).expanduser()
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")

    asm = _load_assembly(mesh_path)
    node_ids, _ = _extract_nodes(asm)
    est_bytes = _estimate_bytes(int(args.timesteps), int(node_ids.shape[0]))
    est_gb = est_bytes / (1024**3)
    print(
        f"[estimate] u/v/a/f_ext memory ~= {est_gb:.3f} GB "
        f"(T={int(args.timesteps)}, N={int(node_ids.shape[0])})"
    )
    if est_gb > float(args.max_gb) and not bool(args.allow_large):
        raise RuntimeError(
            f"Estimated memory {est_gb:.3f} GB exceeds --max-gb={float(args.max_gb):.3f}. "
            "Use smaller --timesteps or pass --allow-large."
        )

    data = _build_template(
        asm=asm,
        timesteps=int(args.timesteps),
        dt=float(args.dt),
        include_edges=bool(args.include_edges),
    )

    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **data)

    schema_path = out_path.with_suffix(out_path.suffix + ".schema.json")
    schema = _schema_for(data)
    with schema_path.open("w", encoding="utf-8") as fp:
        json.dump(schema, fp, ensure_ascii=False, indent=2)

    print(f"[ok] wrote template: {out_path}")
    print(f"[ok] wrote schema  : {schema_path}")
    print(
        "[summary] "
        f"N={data['x0'].shape[0]} T={data['u'].shape[0]} "
        f"edge_E={data['edge_index'].shape[1]}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
