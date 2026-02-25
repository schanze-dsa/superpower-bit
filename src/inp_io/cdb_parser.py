#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cdb_parser.py -- Parse ANSYS CDB into AssemblyModel (nodes/elements/components/contact/bcs).

Scope:
  - NBLOCK / EBLOCK / ETBLOCK
  - Inline APDL ET/CM commands (common in exported CDB)
  - CMBLOCK element components (contact groups + parts)
  - D constraints (node-based DOF fixes)

Notes:
  - SOLID185 is mapped as 8-node hex.
  - Contact/target elements (CONTA173/TARGE170) are kept so contact surfaces
    can be triangulated later.
  - If MIRROR1/MIRROR2 both exist, they are merged into a single "MIRROR" part.
"""
from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Tuple, Iterable

import numpy as np

from inp_io.inp_parser import (
    AssemblyModel,
    PartMesh,
    ElementBlock,
    ContactPair,
    BoundaryEntry,
)
from assembly.surfaces import SurfaceDef

# ----------------------------- helpers -----------------------------


def _parse_fixed_width(line: str, widths: Iterable[int]) -> List[str]:
    out: List[str] = []
    idx = 0
    line = line.rstrip("\n")
    total = sum(widths)
    if len(line) < total:
        line = line + (" " * (total - len(line)))
    for w in widths:
        out.append(line[idx : idx + w].strip())
        idx += w
    return out


def _safe_int(val: str) -> Optional[int]:
    if val is None:
        return None
    s = str(val).strip()
    if not s:
        return None
    try:
        return int(s)
    except Exception:
        try:
            return int(float(s))
        except Exception:
            return None


def _safe_float(val: str) -> Optional[float]:
    if val is None:
        return None
    s = str(val).strip()
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None


def _expand_range_stream(values: List[int]) -> List[int]:
    out: List[int] = []
    last = None
    for v in values:
        if v is None:
            continue
        if v < 0 and last is not None:
            out.extend(range(last, abs(v) + 1))
        else:
            out.append(v)
            last = v
    return out


def _parse_fortran_int_format(fmt_line: str, default_count: int, default_width: int) -> Tuple[int, int]:
    """
    Parse a simple Fortran integer format like "(19i8)" or "(2i9,19a9)".
    Returns (count, width). Falls back to defaults when format is missing/unknown.
    """
    s = (fmt_line or "").strip().lower().replace(" ", "")
    m = re.search(r"(\d+)i(\d+)", s)
    if not m:
        return default_count, default_width
    n = _safe_int(m.group(1))
    w = _safe_int(m.group(2))
    if n is None or n <= 0:
        n = default_count
    if w is None or w <= 0:
        w = default_width
    return int(n), int(w)


def _parse_fortran_float_format(fmt_line: str, default_count: int, default_width: int) -> Tuple[int, int]:
    """
    Parse a simple Fortran float format like "(3i8,6e16.9)".
    Returns (count, width). Falls back to defaults when format is missing/unknown.
    """
    s = (fmt_line or "").strip().lower().replace(" ", "")
    m = re.search(r"(\d+)e(\d+)", s)
    if not m:
        return default_count, default_width
    n = _safe_int(m.group(1))
    w = _safe_int(m.group(2))
    if n is None or n <= 0:
        n = default_count
    if w is None or w <= 0:
        w = default_width
    return int(n), int(w)


def _normalize_component_name(name: str) -> str:
    """Normalize component names for stable matching with config keys."""
    return re.sub(r"\s+", " ", (name or "").strip()).upper()


def _merge_component_ids(existing: List[int], incoming: List[int]) -> List[int]:
    if not existing:
        return sorted({int(v) for v in incoming})
    if not incoming:
        return list(existing)
    return sorted(set(int(v) for v in existing) | set(int(v) for v in incoming))


def _parse_et_command(line: str) -> Tuple[Optional[int], Optional[int]]:
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 3:
        return None, None
    tid = _safe_int(parts[1])
    code = _safe_int(parts[2].split("$")[0].strip())
    return tid, code


def _parse_cm_command(line: str, fallback_idx: int) -> Tuple[str, str]:
    parts = [p.strip() for p in line.split(",")]
    raw_name = parts[1] if len(parts) > 1 and parts[1] else f"CMP_{fallback_idx}"
    ctype = parts[2] if len(parts) > 2 and parts[2] else "ELEM"
    return _normalize_component_name(raw_name), ctype.upper()


# ----------------------------- parse blocks -----------------------------


def _parse_etblock(lines: List[str], start: int) -> Tuple[Dict[int, int], int]:
    # ETBLOCK, n, n
    etype_map: Dict[int, int] = {}
    i = start + 1  # next line is format
    fmt_line = lines[i].strip() if i < len(lines) else ""
    _, int_width = _parse_fortran_int_format(fmt_line, default_count=2, default_width=9)
    i += 1
    while i < len(lines):
        s = lines[i].strip()
        if not s:
            i += 1
            continue
        if s.startswith("-1"):
            i += 1
            break
        # format usually (2i9,19a9) -> two leading integer fields.
        parts = _parse_fixed_width(lines[i], [int_width, int_width])
        tid = _safe_int(parts[0])
        code = _safe_int(parts[1])
        if tid is not None and code is not None:
            etype_map[tid] = code
        i += 1
    return etype_map, i


def _parse_nblock(lines: List[str], start: int) -> Tuple[Dict[int, Tuple[float, float, float]], int]:
    nodes: Dict[int, Tuple[float, float, float]] = {}
    i = start + 1  # format line
    fmt_line = lines[i].strip() if i < len(lines) else ""
    int_count, int_width = _parse_fortran_int_format(fmt_line, default_count=3, default_width=9)
    float_count, float_width = _parse_fortran_float_format(fmt_line, default_count=6, default_width=21)
    i += 1
    int_widths = [int_width] * int_count
    float_widths = [float_width] * float_count
    while i < len(lines):
        s = lines[i].strip()
        if not s:
            i += 1
            continue
        # End markers
        if s.startswith("N,") or s.startswith("EBLOCK") or s.startswith("CMBLOCK") or s.startswith("-1"):
            break
        # Some CDB variants don't emit "-1" and jump directly to next command.
        if s[0].isalpha() or s[0] in {"/", "*"}:
            break
        head = _parse_fixed_width(lines[i], int_widths)
        nid = _safe_int(head[0] if head else "")
        if nid is None:
            i += 1
            continue
        tail = _parse_fixed_width(lines[i][sum(int_widths) :], float_widths)
        x = _safe_float(tail[0])
        y = _safe_float(tail[1])
        z = _safe_float(tail[2])
        if x is None or y is None or z is None:
            i += 1
            continue
        nodes[int(nid)] = (float(x), float(y), float(z))
        i += 1
    return nodes, i


def _parse_eblock(
    lines: List[str],
    start: int,
    etype_map: Dict[int, int],
) -> Tuple[Dict[int, Tuple[str, List[int]]], int]:
    elements: Dict[int, Tuple[str, List[int]]] = {}
    i = start + 1  # format line
    fmt_line = lines[i].strip() if i < len(lines) else ""
    ncols, int_width = _parse_fortran_int_format(fmt_line, default_count=19, default_width=10)
    i += 1
    widths = [int_width] * ncols
    while i < len(lines):
        s = lines[i].strip()
        if not s:
            i += 1
            continue
        if s.startswith("-1"):
            i += 1
            break
        if s.startswith("CMBLOCK") or s.startswith("RLBLOCK"):
            break
        # Common APDL-command boundaries for CM/ET style exports.
        if s[0].isalpha() or s[0] in {"/", "*"}:
            break
        fields = _parse_fixed_width(lines[i], widths)
        ints = [_safe_int(x) for x in fields]
        if len(ints) < 11:
            i += 1
            continue
        elem_id = ints[10]
        type_id = ints[1]
        nnode = ints[8] if ints[8] is not None else 0
        if elem_id is None or type_id is None:
            i += 1
            continue
        node_ids: List[int] = []
        if nnode > 0:
            raw_nodes = ints[11 : 11 + nnode]
            node_ids = [int(n) for n in raw_nodes if n is not None and int(n) != 0]
        code = etype_map.get(int(type_id))
        etype = _etype_name_from_code(code)
        # SOLID186 reduced to first 8 nodes (corner nodes) for DFEM/C3D8 handling.
        if etype == "SOLID186" and len(node_ids) >= 8:
            node_ids = node_ids[:8]
            etype = "SOLID185"
        elements[int(elem_id)] = (etype, node_ids)
        i += 1
    return elements, i


def _parse_cmblock(lines: List[str], start: int) -> Tuple[str, str, List[int], int]:
    # CMBLOCK,NAME,ELEM, <n>
    header = lines[start].strip()
    parts = [p.strip() for p in header.split(",")]
    name = parts[1] if len(parts) > 1 else f"CMP_{start}"
    ctype = parts[2] if len(parts) > 2 else "ELEM"
    i = start + 1  # format line
    fmt_line = ""
    if i < len(lines) and lines[i].strip().startswith("("):
        fmt_line = lines[i].strip()
        i += 1
    data_vals: List[int] = []
    ncols, int_width = _parse_fortran_int_format(fmt_line, default_count=8, default_width=10)
    widths = [int_width] * ncols
    while i < len(lines):
        s = lines[i].strip()
        if not s:
            i += 1
            continue
        if s.startswith("CMBLOCK") or s.startswith("RLBLOCK") or s.startswith("NBLOCK") or s.startswith("EBLOCK"):
            break
        if s.startswith("-1"):
            i += 1
            break
        # parse integer rows according to the CMBLOCK format line.
        fields = _parse_fixed_width(lines[i], widths)
        ints = [_safe_int(x) for x in fields]
        for v in ints:
            if v is not None:
                data_vals.append(int(v))
        i += 1
    ids = _expand_range_stream(data_vals)
    return name, ctype, ids, i


def _etype_name_from_code(code: Optional[int]) -> str:
    if code is None:
        return "UNKNOWN"
    if code == 170:
        return "TARGE170"
    if code == 173:
        return "CONTA173"
    if code == 174:
        return "CONTA174"
    if code == 185:
        return "SOLID185"
    if code == 186:
        return "SOLID186"
    return f"ET_{code}"


def _is_contact_component(name: str) -> Optional[Tuple[str, str]]:
    m = re.match(r"GROUP_TARG_CONT_(\d+)_(MASTER|SLAVE)_COMP", name, re.IGNORECASE)
    if not m:
        return None
    return m.group(1), m.group(2).upper()


def _is_combined_component(name: str, components: Dict[str, List[int]]) -> bool:
    # Do not auto-skip base components (e.g., LUOMU/LUOSHUAN) since they may be
    # distinct parts in the CDB.
    return False


# ----------------------------- main loader -----------------------------


def load_cdb(path: str) -> AssemblyModel:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CDB not found: {path}")

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.read().splitlines()

    etype_map: Dict[int, int] = {}
    nodes: Dict[int, Tuple[float, float, float]] = {}
    elements: Dict[int, Tuple[str, List[int]]] = {}
    components: Dict[str, List[int]] = {}
    component_types: Dict[str, str] = {}
    boundaries: List[BoundaryEntry] = []
    selected_elem_ids: set[int] = set()
    last_eblock_elem_ids: List[int] = []

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line.upper().startswith("ETBLOCK"):
            parsed_map, i = _parse_etblock(lines, i)
            etype_map.update(parsed_map)
            continue
        if line.upper().startswith("ET,"):
            tid, code = _parse_et_command(line)
            if tid is not None and code is not None:
                etype_map[int(tid)] = int(code)
            i += 1
            continue
        if line.upper().startswith("NBLOCK"):
            nodes, i = _parse_nblock(lines, i)
            continue
        if line.upper().startswith("EBLOCK"):
            block_elements, i = _parse_eblock(lines, i, etype_map)
            elements.update(block_elements)
            last_eblock_elem_ids = sorted(block_elements.keys())
            selected_elem_ids = set(last_eblock_elem_ids)
            continue
        if line.upper().startswith("CMBLOCK"):
            name, ctype, ids, i = _parse_cmblock(lines, i)
            cname = _normalize_component_name(name)
            components[cname] = _merge_component_ids(components.get(cname, []), ids)
            component_types[cname] = ctype.upper()
            continue
        if line.upper().startswith("CM,"):
            cname, ctype = _parse_cm_command(line, i)
            if ctype == "ELEM":
                ids = sorted(selected_elem_ids) if selected_elem_ids else list(last_eblock_elem_ids)
                components[cname] = _merge_component_ids(components.get(cname, []), ids)
            else:
                components.setdefault(cname, [])
            component_types[cname] = ctype
            i += 1
            continue
        if line.upper().startswith("ESEL"):
            parts = [p.strip().upper() for p in line.split(",")]
            mode = parts[1] if len(parts) > 1 else ""
            if mode == "NONE":
                selected_elem_ids.clear()
            elif mode == "ALL":
                selected_elem_ids = set(elements.keys())
            i += 1
            continue
        if line.upper().startswith("D,"):
            boundaries.append(BoundaryEntry(raw=line.strip()))
            i += 1
            continue
        i += 1

    model = AssemblyModel()
    model.boundaries = boundaries

    # Build contact pairs from component names
    contact_pairs: List[ContactPair] = []
    contact_groups: Dict[str, Dict[str, str]] = {}
    for name in components.keys():
        hit = _is_contact_component(name)
        if not hit:
            continue
        idx, role = hit
        group = contact_groups.setdefault(idx, {})
        group[role] = name
    for idx, group in sorted(contact_groups.items(), key=lambda kv: int(kv[0])):
        master = group.get("MASTER", "")
        slave = group.get("SLAVE", "")
        if master and slave:
            contact_pairs.append(ContactPair(master=master, slave=slave, interaction=None, raw=""))
    model.contact_pairs = contact_pairs

    # Build parts from component sets (skip contact components)
    part_components: Dict[str, List[int]] = {}
    for name, ids in components.items():
        if _is_contact_component(name):
            continue
        if _is_combined_component(name, components):
            continue
        if component_types.get(name, "") != "ELEM":
            continue
        part_components[name] = ids

    # Merge mirror parts if both exist
    mirror1 = next((n for n in part_components if n.upper() == "MIRROR1"), None)
    mirror2 = next((n for n in part_components if n.upper() == "MIRROR2"), None)
    if mirror1 and mirror2:
        merged = sorted(set(part_components[mirror1]) | set(part_components[mirror2]))
        part_components["MIRROR"] = merged
        del part_components[mirror1]
        del part_components[mirror2]

    # Create parts
    for name, elem_ids in part_components.items():
        part = PartMesh(name=name)
        blocks: Dict[str, Tuple[List[int], List[List[int]]]] = {}
        for eid in elem_ids:
            if eid not in elements:
                continue
            etype, conn = elements[eid]
            if etype.startswith("CONTA") or etype.startswith("TARGE"):
                continue
            blk = blocks.setdefault(etype, ([], []))
            blk[0].append(int(eid))
            blk[1].append([int(n) for n in conn])
        for etype, (eids, conns) in blocks.items():
            part.element_blocks.append(ElementBlock(etype, eids, conns, {}))

        # nodes for part
        node_ids = set()
        for blk in part.element_blocks:
            for conn in blk.connectivity:
                for nid in conn:
                    node_ids.add(int(nid))
        part.node_ids = sorted(node_ids)
        part.nodes_xyz = {nid: nodes[nid] for nid in part.node_ids if nid in nodes}
        model.parts[name] = part

    # Contact elements in a separate part for surface triangulation
    contact_elem_ids = [
        eid for eid, (etype, _) in elements.items()
        if etype.startswith("CONTA") or etype.startswith("TARGE")
    ]
    if contact_elem_ids:
        part = PartMesh(name="__CONTACT__")
        blocks: Dict[str, Tuple[List[int], List[List[int]]]] = {}
        for eid in contact_elem_ids:
            etype, conn = elements[eid]
            blk = blocks.setdefault(etype, ([], []))
            blk[0].append(int(eid))
            blk[1].append([int(n) for n in conn])
        for etype, (eids, conns) in blocks.items():
            part.element_blocks.append(ElementBlock(etype, eids, conns, {}))
        node_ids = set()
        for blk in part.element_blocks:
            for conn in blk.connectivity:
                for nid in conn:
                    node_ids.add(int(nid))
        part.node_ids = sorted(node_ids)
        part.nodes_xyz = {nid: nodes[nid] for nid in part.node_ids if nid in nodes}
        model.parts[part.name] = part

    # Surface definitions from components (element sets)
    for name, ids in components.items():
        if component_types.get(name, "") != "ELEM":
            continue
        items = [(int(eid), "") for eid in ids if eid in elements]
        model.surfaces[name] = SurfaceDef(
            stype="ELEMENT",
            name=name,
            items=items,
            owner=None,
            scope="assembly",
            raw_lines=None,
        )

    # Convenience: create a mirror-up alias for visualization when mirror part exists.
    mirror_part = None
    for cand in ("MIRROR", "MIRROR1"):
        if cand in model.parts:
            mirror_part = cand
            break
    if mirror_part:
        # Use one element to seed part resolution; viz can rebuild part_top surface later.
        try:
            part = model.parts[mirror_part]
            seed_eid = None
            for blk in part.element_blocks:
                if blk.elem_ids:
                    seed_eid = int(blk.elem_ids[0])
                    break
            if seed_eid is not None:
                model.surfaces["MIRROR UP"] = SurfaceDef(
                    stype="ELEMENT",
                    name="MIRROR UP",
                    items=[(seed_eid, "S1")],
                    owner=mirror_part,
                    scope="part",
                    raw_lines=None,
                )
        except Exception:
            pass

    # Fill nodes/elements at assembly level
    model.nodes = nodes
    model.elements = {eid: conn for eid, (_, conn) in elements.items()}
    return model


def main():
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Parse ANSYS .cdb")
    ap.add_argument("--cdb", type=str, required=True)
    ap.add_argument("--dump_json", type=str, default="")
    args = ap.parse_args()
    asm = load_cdb(args.cdb)
    print(asm.summary())
    if args.dump_json:
        with open(args.dump_json, "w", encoding="utf-8") as f:
            json.dump(asm.summary(), f, ensure_ascii=False, indent=2)
        print(f"Summary JSON saved to: {args.dump_json}")


if __name__ == "__main__":
    main()
