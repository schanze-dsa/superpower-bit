#!/usr/bin/env python3
"""
Generate a 100-paper Chinese brief for static/quasi-static deformation prediction.

Strategy:
1) Pull papers from a curated high-tier journal list via OpenAlex source IDs.
2) Keep 2019+ article papers with DOI + abstract.
3) Filter by deformation/structural mechanics relevance.
4) Rank and export:
   - notes/100篇高分区文献要点与创新凝练.txt
   - notes/_100papers_high_tier_openalex.json
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import requests


OPENALEX_WORKS_URL = "https://api.openalex.org/works"
OPENALEX_SOURCES_URL = "https://api.openalex.org/sources"


# Journal pool focused on structural mechanics / computational mechanics / engineering AI.
TARGET_JOURNALS = [
    "Computer Methods in Applied Mechanics and Engineering",
    "International Journal for Numerical Methods in Engineering",
    "Computational Mechanics",
    "International Journal of Solids and Structures",
    "International Journal of Mechanical Sciences",
    "International Journal of Engineering Science",
    "Mechanics of Materials",
    "Engineering Structures",
    "Composite Structures",
    "Thin-Walled Structures",
    "Structural and Multidisciplinary Optimization",
    "Computer-Aided Civil and Infrastructure Engineering",
    "Engineering with Computers",
    "Smart Materials and Structures",
    "Mechanical Systems and Signal Processing",
    "Journal of Sound and Vibration",
    "Reliability Engineering & System Safety",
    "Applied Mathematical Modelling",
    "Engineering Applications of Artificial Intelligence",
    "Advanced Engineering Informatics",
    "Automation in Construction",
    "Computers and Structures",
    "European Journal of Mechanics - A/Solids",
    "Archives of Computational Methods in Engineering",
    "npj Computational Materials",
]


JOURNAL_POOL = set(TARGET_JOURNALS)


MECH_TERMS = [
    "deformation",
    "displacement",
    "elastic",
    "elasto",
    "solid mechanics",
    "structural",
    "finite element",
    "strain",
    "stress",
    "contact",
    "constitutive",
    "buckling",
    "deflection",
]


METHOD_TERMS = [
    "neural",
    "machine learning",
    "data-driven",
    "surrogate",
    "operator learning",
    "neural operator",
    "graph",
    "pinn",
    "physics informed",
    "deep learning",
    "reduced order",
    "gaussian process",
]


NOISE_TITLE_TERMS = [
    "review",
    "survey",
    "bibliometric",
    "state of the art",
    "special issue",
    "editorial",
]


METHOD_KEYWORDS = [
    ("physics informed neural network", "PINN"),
    ("pinn", "PINN"),
    ("graph neural network", "GNN"),
    ("message passing", "图消息传递"),
    ("neural operator", "神经算子"),
    ("deep operator", "DeepONet/算子学习"),
    ("deep learning", "深度学习"),
    ("transformer", "Transformer"),
    ("surrogate", "代理模型"),
    ("reduced order", "降阶模型"),
    ("gaussian process", "高斯过程"),
]


TASK_KEYWORDS = [
    ("static", "静态工况"),
    ("quasi-static", "准静态工况"),
    ("deformation", "变形预测"),
    ("displacement", "位移场预测"),
    ("elasticity", "弹性力学问题"),
    ("solid mechanics", "固体力学问题"),
    ("contact", "接触力学"),
    ("finite element", "有限元加速"),
    ("structural", "结构响应"),
    ("inverse", "反问题识别"),
]


VALUE_KEYWORDS = [
    ("accuracy", "精度提升"),
    ("efficient", "效率提升"),
    ("real-time", "实时推断"),
    ("generalization", "泛化能力"),
    ("uncertainty", "不确定性量化"),
    ("robust", "鲁棒性"),
    ("multi-scale", "多尺度建模"),
]


RELEVANCE_WEIGHTS = {
    "deformation": 5.0,
    "displacement": 4.5,
    "deflection": 4.5,
    "structural": 3.5,
    "solid mechanics": 4.0,
    "elasticity": 4.0,
    "finite element": 5.0,
    "contact": 5.5,
    "surrogate": 5.0,
    "physics informed": 6.0,
    "pinn": 6.0,
    "graph neural": 4.5,
    "neural operator": 4.5,
    "reduced order": 4.0,
    "static": 3.0,
    "quasi-static": 3.0,
}


@dataclass
class Paper:
    doi: str
    title: str
    year: int
    journal: str
    cited_by: int
    abstract: str
    openalex_id: str
    relevance: float
    source_id: str


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def invert_abstract(inv: Dict[str, List[int]] | None) -> str:
    if not inv:
        return ""
    max_pos = -1
    for positions in inv.values():
        if positions:
            max_pos = max(max_pos, max(positions))
    if max_pos < 0:
        return ""
    tokens = [""] * (max_pos + 1)
    for word, positions in inv.items():
        for pos in positions:
            if 0 <= pos < len(tokens):
                tokens[pos] = word
    return " ".join(t for t in tokens if t).strip()


def detect_terms(text: str, rules: Iterable[Tuple[str, str]], k: int = 2) -> List[str]:
    low = text.lower()
    hits: List[str] = []
    for key, label in rules:
        if key in low and label not in hits:
            hits.append(label)
        if len(hits) >= k:
            break
    return hits


def short_keypoint(title: str, abstract: str) -> str:
    text = f"{title}. {abstract}".strip()
    methods = detect_terms(text, METHOD_KEYWORDS, k=2)
    tasks = detect_terms(text, TASK_KEYWORDS, k=2)
    values = detect_terms(text, VALUE_KEYWORDS, k=2)
    m = "、".join(methods) if methods else "数据驱动方法"
    t = "、".join(tasks) if tasks else "结构变形相关任务"
    v = "、".join(values) if values else "精度与工程可用性"
    focus = title.split(":")[0].strip()
    if len(focus) > 42:
        focus = focus[:42].rstrip() + "..."
    return f"围绕“{focus}”问题，面向{t}采用{m}；重点关注{v}。"


def relevance_score(title: str, abstract: str, cited_by: int) -> float:
    text = f"{title} {abstract}".lower()
    score = 0.0
    for key, weight in RELEVANCE_WEIGHTS.items():
        if key in text:
            score += weight
    # Static/quasi-static preference for current user goal.
    has_static = ("static" in text) or ("quasi-static" in text) or ("elastostat" in text)
    has_dynamic = ("dynamic" in text) or ("vibration" in text) or ("transient" in text)
    if has_static:
        score += 2.0
    if has_dynamic and not has_static:
        score -= 2.0
    # Keep citation effect moderate.
    score += min(cited_by, 600) * 0.015
    return score


def is_target_paper(title: str, abstract: str) -> bool:
    title_low = title.lower()
    if any(word in title_low for word in NOISE_TITLE_TERMS):
        return False
    text = f"{title} {abstract}".lower()
    has_static = ("static" in text) or ("quasi-static" in text) or ("elastostat" in text)
    has_dynamic = ("dynamic" in text) or ("vibration" in text) or ("seismic" in text)
    if has_dynamic and not has_static:
        # Current task is static/quasi-static deformation.
        return False
    mech_hits = sum(word in text for word in MECH_TERMS)
    method_hits = sum(word in text for word in METHOD_TERMS)
    if mech_hits >= 2 and method_hits >= 1:
        return True
    if mech_hits >= 4:
        return True
    return False


def get_json_with_retry(session: requests.Session, url: str, params: Dict[str, str], retries: int = 3) -> dict:
    for i in range(retries):
        try:
            r = session.get(url, params=params, timeout=40)
            r.raise_for_status()
            return r.json()
        except Exception:
            if i == retries - 1:
                raise
            time.sleep(1.0 + i * 1.2)
    raise RuntimeError("unreachable")


def resolve_source_ids(journal_names: List[str]) -> Dict[str, Tuple[str, str]]:
    """
    Return mapping:
    target_journal_name -> (source_id, resolved_display_name)
    """
    session = requests.Session()
    resolved: Dict[str, Tuple[str, str]] = {}
    for name in journal_names:
        params = {"search": name, "per-page": 25}
        data = get_json_with_retry(session, OPENALEX_SOURCES_URL, params=params)
        results = data.get("results", [])
        if not results:
            continue
        name_low = normalize_space(name).lower()

        chosen = None
        for s in results:
            dname = normalize_space(s.get("display_name", ""))
            if dname.lower() == name_low:
                chosen = s
                break
        if chosen is None:
            for s in results:
                dname = normalize_space(s.get("display_name", ""))
                if name_low in dname.lower() or dname.lower() in name_low:
                    chosen = s
                    break
        if chosen is None:
            chosen = results[0]

        sid = chosen.get("id", "")
        dname = normalize_space(chosen.get("display_name", ""))
        if sid and dname:
            resolved[name] = (sid, dname)
        time.sleep(0.08)
    return resolved


def fetch_works_for_source(source_id: str, max_pages: int, per_page: int = 120) -> List[dict]:
    session = requests.Session()
    out: List[dict] = []
    for page in range(1, max_pages + 1):
        params = {
            "filter": ",".join(
                [
                    "from_publication_date:2019-01-01",
                    "type:article",
                    "has_doi:true",
                    "has_abstract:true",
                    f"primary_location.source.id:{source_id}",
                ]
            ),
            "sort": "cited_by_count:desc",
            "per-page": str(per_page),
            "page": str(page),
        }
        data = get_json_with_retry(session, OPENALEX_WORKS_URL, params=params)
        batch = data.get("results", [])
        if not batch:
            break
        out.extend(batch)
        time.sleep(0.12)
    return out


def build_candidates(max_pages_per_journal: int) -> Tuple[List[Paper], Dict[str, Tuple[str, str]]]:
    resolved_sources = resolve_source_ids(TARGET_JOURNALS)
    doi_map: Dict[str, Paper] = {}

    for target_name, (source_id, resolved_name) in resolved_sources.items():
        works = fetch_works_for_source(source_id, max_pages=max_pages_per_journal)
        for w in works:
            doi = normalize_space(w.get("doi") or "").lower()
            if not doi:
                continue

            title = normalize_space(w.get("display_name") or "")
            year = int(w.get("publication_year") or 0)
            cited_by = int(w.get("cited_by_count") or 0)
            source_name = normalize_space(
                ((w.get("primary_location") or {}).get("source") or {}).get("display_name", "")
            )
            abstract = invert_abstract(w.get("abstract_inverted_index"))

            if not title or not source_name or not abstract:
                continue
            if source_name not in JOURNAL_POOL and source_name != resolved_name:
                # Keep only intended journals.
                continue
            if not is_target_paper(title, abstract):
                continue

            score = relevance_score(title, abstract, cited_by)
            paper = Paper(
                doi=doi,
                title=title,
                year=year,
                journal=source_name,
                cited_by=cited_by,
                abstract=abstract,
                openalex_id=normalize_space(w.get("id") or ""),
                relevance=score,
                source_id=source_id,
            )
            old = doi_map.get(doi)
            if old is None or paper.relevance > old.relevance:
                doi_map[doi] = paper
    return list(doi_map.values()), resolved_sources


def select_top_100(candidates: List[Paper], min_relevance: float, max_per_journal: int) -> List[Paper]:
    filtered: List[Paper] = []
    for p in candidates:
        text = f"{p.title} {p.abstract}".lower()
        has_static = ("static" in text) or ("quasi-static" in text) or ("elastostat" in text)
        has_dynamic = ("dynamic" in text) or ("vibration" in text) or ("seismic" in text)
        if has_dynamic and not has_static:
            continue
        if p.relevance < min_relevance:
            continue
        filtered.append(p)
    ranked = sorted(filtered, key=lambda p: (p.relevance, p.cited_by, p.year), reverse=True)

    selected: List[Paper] = []
    cnt = Counter()
    for p in ranked:
        if cnt[p.journal] >= max_per_journal:
            continue
        selected.append(p)
        cnt[p.journal] += 1
        if len(selected) >= 100:
            return selected

    if len(selected) < 100:
        used = {p.doi for p in selected}
        for p in ranked:
            if p.doi in used:
                continue
            selected.append(p)
            used.add(p.doi)
            if len(selected) >= 100:
                return selected
    return selected


def build_innovation_section(selected: List[Paper]) -> str:
    top_journals = Counter(p.journal for p in selected).most_common(6)
    journal_hint = "、".join(j for j, _ in top_journals) if top_journals else "高水平结构力学期刊"
    lines = [
        "面向你的目标（静态/准静态变形预测）的创新点凝练",
        "------------------------------------------------",
        "以下创新点按“新 + 有用 + 可证 + 可复现”组织，可直接映射到论文的方法与实验章节：",
        "",
        "创新点 1：区域感知双分支 MLP-PINN（全局趋势 + 接触残差）",
        "1) 主干 MLP 预测全域位移场，局部 MLP 仅在接触/孔边/镜面区域学习残差。",
        "2) 融合方式：u = u_global + mask * u_local_residual。",
        "3) 优势：同时兼顾全局平滑性和关键区域精度。",
        "",
        "创新点 2：分层物理约束课程训练（Global -> Contact）",
        "1) 前期强化全域平衡/能量一致性，后期逐步提高接触一致性和区域误差权重。",
        "2) 通过阶段化权重避免训练初期接触项主导导致不稳定。",
        "",
        "创新点 3：CDB 先验特征注入（几何-物理联合编码）",
        "1) 从 CDB 自动提取边界距离、接触面法向、单元尺度、材料编码等输入特征。",
        "2) 不改变主干 MLP 结构，只增强输入信息密度，便于工程部署。",
        "",
        "创新点 4：工程导向评价体系（关键区域优先）",
        "1) 指标组合：全局 RMSE + 镜面区 RMSE + 接触区 MAE + 最大误差 + 95%分位误差。",
        "2) 同时报推理耗时与 FEM 加速比，支撑工程价值。",
        "",
        "建议写入论文的可证伪阈值：",
        "1) 接触区 MAE 相比单分支基线下降 >= 12%。",
        "2) 镜面区 RMSE 下降 >= 10%，全局 RMSE 变化 <= +2%。",
        "3) 推理速度相对 FEM 加速 >= 20x（同硬件设置）。",
        "",
        f"文献分布提示：本次100篇主要来自 {journal_hint}，热点集中于 PINN/GNN/神经算子/代理模型。",
    ]
    return "\n".join(lines)


def export_outputs(
    selected: List[Paper],
    resolved_sources: Dict[str, Tuple[str, str]],
    out_txt: Path,
    out_json: Path,
    cache_file: Path,
) -> None:
    out_txt.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, p in enumerate(selected, start=1):
        rows.append(
            {
                "rank": i,
                "title": p.title,
                "journal": p.journal,
                "year": p.year,
                "doi": p.doi,
                "cited_by": p.cited_by,
                "relevance": round(p.relevance, 3),
                "keypoint_cn": short_keypoint(p.title, p.abstract),
                "openalex_id": p.openalex_id,
                "source_id": p.source_id,
                "abstract": p.abstract,
            }
        )
    out_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    lines: List[str] = []
    lines.append("静态变形预测：100篇高分区文献要点与创新凝练（自动检索版）")
    lines.append("================================================================")
    lines.append("")
    lines.append("检索口径")
    lines.append("--------")
    lines.append("1) 数据源：OpenAlex（执行时实时检索）。")
    lines.append("2) 论文类型：2019年后 Article，含 DOI 与摘要。")
    lines.append("3) 先按高水平期刊池筛选，再按结构变形任务相关关键词过滤。")
    lines.append("4) 分区说明：按常见中科院一区期刊池构建，正式投稿前请用学校分区工具逐篇复核。")
    lines.append("5) 缓存文件：{}".format(cache_file.as_posix()))
    lines.append("")
    lines.append("检索到的目标期刊与 OpenAlex Source")
    lines.append("-----------------------------------")
    for k in TARGET_JOURNALS:
        if k in resolved_sources:
            sid, dname = resolved_sources[k]
            lines.append(f"- {k} -> {dname} ({sid})")
        else:
            lines.append(f"- {k} -> [未解析到]")
    lines.append("")
    lines.append(build_innovation_section(selected))
    lines.append("")
    lines.append("100篇论文要点")
    lines.append("-------------")
    for row in rows:
        lines.append(f"{row['rank']}. {row['title']}")
        lines.append(
            f"   期刊：{row['journal']} | 年份：{row['year']} | 被引：{row['cited_by']} | DOI：{row['doi']}"
        )
        lines.append(f"   要点：{row['keypoint_cn']}")
        lines.append("")
    out_txt.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-pages-per-journal", type=int, default=2)
    parser.add_argument("--min-relevance", type=float, default=3.0)
    parser.add_argument("--max-per-journal", type=int, default=8)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument(
        "--cache-file",
        type=Path,
        default=Path("notes/_openalex_candidates_cache.json"),
    )
    parser.add_argument(
        "--sources-cache-file",
        type=Path,
        default=Path("notes/_openalex_sources_cache.json"),
    )
    parser.add_argument(
        "--out-txt",
        type=Path,
        default=Path("notes/100篇高分区文献要点与创新凝练.txt"),
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("notes/_100papers_high_tier_openalex.json"),
    )
    args = parser.parse_args()

    if args.cache_file.exists() and args.sources_cache_file.exists() and not args.refresh_cache:
        raw = json.loads(args.cache_file.read_text(encoding="utf-8"))
        candidates = [Paper(**x) for x in raw]
        resolved_sources = {
            k: (v["source_id"], v["resolved_name"])
            for k, v in json.loads(args.sources_cache_file.read_text(encoding="utf-8")).items()
        }
    else:
        candidates, resolved_sources = build_candidates(args.max_pages_per_journal)
        args.cache_file.parent.mkdir(parents=True, exist_ok=True)
        args.cache_file.write_text(
            json.dumps([c.__dict__ for c in candidates], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        args.sources_cache_file.write_text(
            json.dumps(
                {
                    k: {"source_id": sid, "resolved_name": dname}
                    for k, (sid, dname) in resolved_sources.items()
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    selected = select_top_100(
        candidates=candidates,
        min_relevance=args.min_relevance,
        max_per_journal=args.max_per_journal,
    )
    if len(selected) < 100:
        raise RuntimeError(
            f"Only selected {len(selected)} papers. "
            "Try increasing --max-pages-per-journal, lowering --min-relevance, or broadening TARGET_JOURNALS."
        )

    export_outputs(
        selected=selected,
        resolved_sources=resolved_sources,
        out_txt=args.out_txt,
        out_json=args.out_json,
        cache_file=args.cache_file,
    )

    print(f"[ok] wrote: {args.out_txt}")
    print(f"[ok] wrote: {args.out_json}")
    print(f"[stats] candidates={len(candidates)}, selected={len(selected)}")


if __name__ == "__main__":
    main()
