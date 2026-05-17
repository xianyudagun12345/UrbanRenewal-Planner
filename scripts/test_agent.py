"""
Agent 端到端测试脚本。

用法：
    python scripts/test_agent.py
    python scripts/test_agent.py --question "控江路和本溪路路口有哪些步行环境问题？"
    python scripts/test_agent.py --question "..." --streetview   # 强制启用街景分析

输出完整 Agent 运行过程（含各节点耗时）和最终规划建议。
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# 设置详细日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_agent")

from src.urbanrenewal.agent.planner import build_graph, PlannerState
from src.urbanrenewal.config import cfg

DEFAULT_QUESTION = "请分析鞍山新村周边800米的老年友好问题"

DIVIDER = "=" * 70


def print_state_summary(state: dict) -> None:
    """打印各节点收集到的数据摘要。"""
    print(f"\n{DIVIDER}")
    print("  Agent 中间状态摘要")
    print(DIVIDER)

    intent = state.get("intent", {})
    if intent:
        print(f"[意图解析]")
        print(f"  地点     : {intent.get('place')}")
        print(f"  场景     : {intent.get('scenario')}")
        print(f"  半径     : {intent.get('radius_m')}m")
        print(f"  需要街景 : {intent.get('need_streetview')}")
        print(f"  超出范围 : {intent.get('out_of_scope')}")

    geo = state.get("geocode_result")
    if geo:
        print(f"\n[地理编码]")
        print(f"  地址     : {geo.address}")
        print(f"  WGS84    : ({geo.lon_wgs84:.5f}, {geo.lat_wgs84:.5f})")
        print(f"  杨浦区   : {geo.in_target_district}")

    poi_summary = state.get("poi_summary") or []
    if poi_summary:
        print(f"\n[POI 统计] 共 {len(poi_summary)} 类设施")
        for row in poi_summary[:8]:
            print(f"  {row['category_planning']:<10} {row['count']:>4} 处  最近 {row['min_distance_m']}m")

    gap = state.get("gap_report")
    if gap:
        print(f"\n[设施缺口]")
        if gap.missing:
            print(f"  完全缺失 : {', '.join(gap.missing)}")
        if gap.insufficient:
            print(f"  数量不足 : {', '.join(gap.insufficient)}")
        if gap.present:
            print(f"  已配置   : {', '.join(gap.present[:5])}{'…' if len(gap.present) > 5 else ''}")

    iso = state.get("isochrone")
    if iso:
        print(f"\n[路网分析]")
        print(f"  步行10分钟等时圈节点 : {iso.node_count} 个")

    intersections = state.get("intersections") or []
    if intersections:
        print(f"  路口（min_street=3）  : {len(intersections)} 个")
        for r in intersections[:3]:
            print(f"    ({r.lon:.5f},{r.lat:.5f}) street_count={r.street_count} {r.distance_m}m")

    sv_results = state.get("sv_results") or []
    if sv_results:
        print(f"\n[街景分析] {len(sv_results)} 张")
        for r in sv_results:
            cache_flag = "[缓存]" if r.from_cache else "[新分析]"
            preview = (r.analysis_text or "")[:60].replace("\n", " ")
            print(f"  {r.image_id}  {r.meta.direction_bucket}向  {r.meta.distance_m}m  {cache_flag}")
            if preview:
                print(f"    {preview}…")

    policy_chunks = state.get("policy_chunks") or []
    if policy_chunks:
        print(f"\n[政策检索] {len(policy_chunks)} 条 chunk")
        seen_docs: set[str] = set()
        for c in policy_chunks[:6]:
            if c.doc_name not in seen_docs:
                print(f"  [{c.score:.3f}] {c.doc_name[:40]}")
                seen_docs.add(c.doc_name)

    error = state.get("error", "")
    if error:
        print(f"\n[错误] {error}")


def run_with_timing(question: str, force_streetview: bool = False) -> None:
    print(DIVIDER)
    print(f"  UrbanRenewal Planner Agent — {cfg.city}{cfg.district}")
    print(DIVIDER)
    print(f"问题：{question}\n")

    graph = build_graph()
    initial: PlannerState = {
        "question": question,
        "intent": {},
        "geocode_result": None,
        "poi_summary": None,
        "gap_report": None,
        "isochrone": None,
        "intersections": [],
        "sv_results": [],
        "policy_chunks": [],
        "answer": "",
        "error": "",
    }

    # 若强制街景，在 intent 中提前注入（测试用）
    if force_streetview:
        initial["intent"] = {"need_streetview": True}

    t0 = time.time()
    final_state = graph.invoke(initial)
    elapsed = time.time() - t0

    # 打印中间状态摘要
    print_state_summary(final_state)

    # 打印最终建议
    print(f"\n{DIVIDER}")
    print("  最终规划建议")
    print(DIVIDER)
    answer = final_state.get("answer", "（无输出）")
    print(answer)

    print(f"\n{DIVIDER}")
    print(f"  总耗时：{elapsed:.1f}s")
    print(DIVIDER)


def main() -> None:
    parser = argparse.ArgumentParser(description="Agent 端到端测试")
    parser.add_argument("--question", "-q", type=str, default=DEFAULT_QUESTION,
                        help=f"测试问题（默认：{DEFAULT_QUESTION}）")
    parser.add_argument("--streetview", action="store_true",
                        help="强制开启街景分析（覆盖意图解析结果）")
    args = parser.parse_args()
    run_with_timing(args.question, force_streetview=args.streetview)


if __name__ == "__main__":
    main()
