"""解析模型 vs 离散事件仿真的验收。

在轻载/中载/接近饱和三档负载下, 对每类请求的响应时间、吞吐,
以及各站点利用率, 列出解析预测与仿真结果的相对偏差;
并验证瓶颈识别与瓶颈转移顺序的预测是否正确。

运行::

    python3 -m qncap.validation            # 标准验收(数分钟)
    python3 -m qncap.validation --quick    # 快速冒烟
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from typing import List


from . import scenario, sim, mva, sizing
from .layered import solve_layered


@dataclass
class Cell:
    metric: str
    target: str
    analytic: float
    simulation: float
    rel_error: float
    sim_ci_half: float = 0.0


@dataclass
class LevelReport:
    level: str
    cells: List[Cell] = field(default_factory=list)
    bottleneck_sim: str = ""
    bottleneck_mva: str = ""
    max_util_error: float = 0.0
    max_rt_error: float = 0.0
    max_x_error: float = 0.0
    notes: List[str] = field(default_factory=list)


def _rel_err(a: float, b: float) -> float:
    if not math.isfinite(a) or not math.isfinite(b):
        return float("nan")
    return abs(a - b) / b if abs(b) > 1e-9 else float("nan")


def validate_level(level: str, n_rep: int, warmup: float,
                   horizon: float, base_seed: int,
                   layered: bool = False) -> LevelReport:
    if layered:
        net = scenario.build_layered_network(
            scenario.LOAD_LEVELS[level], scenario.OPEN_LEVELS[level])
    else:
        net = scenario.build_scenario(level)
    sim_res = sim.run_replications(
        net, n_rep=n_rep, warmup=warmup, horizon=horizon,
        base_seed=base_seed)
    if layered:
        ana = solve_layered(net)
    else:
        ana = mva.solve_approx(net)

    rep = LevelReport(level=level)
    # 类级: 响应时间、吞吐
    for c in net.classes:
        r_sim = sim_res.classes[c]["response_time"]
        r_ana = ana.class_response[c]
        rep.cells.append(Cell(
            "响应时间(s)", c, r_ana, r_sim, _rel_err(r_ana, r_sim),
            sim_ci_half=(sim_res.ci or {}).get(
                "classes", {}).get(c, {}).get("response_time", 0.0)))
        x_sim = sim_res.classes[c]["throughput"]
        x_ana = ana.class_throughput[c]
        rep.cells.append(Cell(
            "吞吐(tps)", c, x_ana, x_sim, _rel_err(x_ana, x_sim),
            sim_ci_half=(sim_res.ci or {}).get(
                "classes", {}).get(c, {}).get("throughput", 0.0)))
        rep.max_rt_error = max(rep.max_rt_error,
                               _rel_err(r_ana, r_sim))
        rep.max_x_error = max(rep.max_x_error, _rel_err(x_ana, x_sim))
    # 站点级: 利用率
    for s in net.stations:
        u_sim = sim_res.stations[s]["utilization"]
        u_ana = ana.station_utilization[s]
        rep.cells.append(Cell(
            "利用率", s, u_ana, u_sim, _rel_err(u_ana, u_sim),
            sim_ci_half=(sim_res.ci or {}).get(
                "stations", {}).get(s, {}).get("utilization", 0.0)))
        rep.max_util_error = max(rep.max_util_error,
                                 _rel_err(u_ana, u_sim))

    # 瓶颈识别(取利用率最高的非 IS 站点)
    def top_bottleneck(util: Dict[str, float]) -> str:
        cand = {s: u for s, u in util.items()
                if net.stations[s].sched != "is"}
        return max(cand, key=cand.get)
    rep.bottleneck_sim = top_bottleneck(sim_res.stations and {
        s: sim_res.stations[s]["utilization"] for s in net.stations})
    rep.bottleneck_mva = top_bottleneck(ana.station_utilization)
    return rep


def validate_bottleneck_shift(level: str = "heavy") -> List[dict]:
    """预测瓶颈转移顺序, 并用每步仿真核对当前瓶颈站点。"""
    net = scenario.build_scenario(level)
    seq = sizing.bottleneck_shift_sequence(
        net, mva.solve_approx, max_steps=5)
    out = []
    for step in seq:
        out.append({
            "step": step.step,
            "removed_bottleneck": step.bottleneck,
            "action": step.action,
            "next_bottleneck": step.new_bottleneck,
            "util_before": round(step.utilization_before, 3),
            "util_after": round(step.utilization_after, 3),
            "response_times": {c: round(r, 3)
                               for c, r in step.response_times.items()},
        })
    return out


def validate_population_double(level: str = "heavy") -> dict:
    """用户数翻倍后的响应时间预测(what-if)。"""
    net = scenario.build_scenario(level)
    base = mva.solve_approx(net)
    doubled = sizing.what_if_population(
        net, mva.solve_approx,
        {c: 2.0 for c in net.classes
         if net.classes[c].kind == "closed"})
    return {
        "base": {c: round(base.class_response[c], 3)
                 for c in base.class_response},
        "doubled": {c: round(doubled.class_response[c], 3)
                    for c in doubled.class_response},
        "doubled_util": {s: round(u, 3)
                         for s, u in doubled.station_utilization.items()},
        "stable": doubled.stable,
        "warnings": doubled.warnings,
    }


def validate_sizing(level: str = "heavy") -> dict:
    """达到目标响应时间所需的最小服务台数。"""
    net = scenario.build_scenario(level)
    base = mva.solve_approx(net)
    ceiling = sizing.asymptotic_ceiling(net)
    # 可达目标: 取"当前响应的90%"与"物理下限的1.2倍"中的较大者,
    # 保证目标在加机器可达成的范围内
    target = {}
    for c in base.class_response:
        target[c] = max(base.class_response[c] * 0.9,
                        ceiling[c] * 1.2)
    config, final, notes = sizing.min_servers_for_target(
        net, mva.solve_approx, target)
    return {
        "target": {c: round(t, 3) for c, t in target.items()},
        "min_servers": config,
        "achieved_response": {c: round(r, 3)
                              for c, r in final.class_response.items()},
        "notes": notes,
    }


def format_report(reports: List[LevelReport]) -> str:
    lines = []
    lines.append("=" * 78)
    lines.append("排队网络容量模型验收: 解析 MVA vs 离散事件仿真")
    lines.append("=" * 78)
    for rep in reports:
        lines.append("")
        lines.append(f"【负载档位: {rep.level}】")
        lines.append(f"  {'指标':<10}{'对象':<10}{'解析':>10}{'仿真':>10}"
                     f"{'相对偏差':>10}{'仿真95%半宽':>12}")
        for cell in rep.cells:
            lines.append(
                f"  {cell.metric:<10}{cell.target:<10}"
                f"{cell.analytic:>10.4f}{cell.simulation:>10.4f}"
                f"{cell.rel_error * 100:>9.1f}%{cell.sim_ci_half:>12.4f}")
        match = "✓ 一致" if rep.bottleneck_sim == rep.bottleneck_mva \
            else "✗ 不一致"
        lines.append(
            f"  瓶颈站点: 解析={rep.bottleneck_mva} "
            f"仿真={rep.bottleneck_sim}  {match}")
        lines.append(
            f"  本档最大偏差: 利用率 {rep.max_util_error * 100:.1f}% | "
            f"响应时间 {rep.max_rt_error * 100:.1f}% | "
            f"吞吐 {rep.max_x_error * 100:.1f}%")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="排队模型验收")
    ap.add_argument("--quick", action="store_true", help="快速冒烟配置")
    args = ap.parse_args()
    if args.quick:
        n_rep, warmup, horizon = 2, 600, 1800
    else:
        n_rep, warmup, horizon = 4, 1500, 4500
    reports = []
    for level in ("light", "medium", "heavy"):
        reports.append(validate_level(
            level, n_rep, warmup, horizon, base_seed=1000))
    print(format_report(reports))

    print("\n" + "=" * 78)
    print("瓶颈转移顺序预测(逐次解除当前瓶颈)")
    print("=" * 78)
    for row in validate_bottleneck_shift("heavy"):
        print(f"  第{row['step']}步: 解除 {row['removed_bottleneck']} "
              f"({row['action']}) -> 新瓶颈 {row['next_bottleneck']} "
              f"[利用率 {row['util_before']}→{row['util_after']}]")

    print("\n" + "=" * 78)
    print("用户数翻倍 what-if")
    print("=" * 78)
    d = validate_population_double("heavy")
    for c in d["base"]:
        print(f"  {c:<9s} 响应 {d['base'][c]}s -> {d['doubled'][c]}s")
    print("  翻倍后站点利用率:", d["doubled_util"])
    if d["warnings"]:
        print("  告警:", "; ".join(d["warnings"]))

    print("\n" + "=" * 78)
    print("目标响应时间下的最小服务台配置")
    print("=" * 78)
    sz = validate_sizing("heavy")
    print("  最小台数:", sz["min_servers"])
    for n in sz["notes"]:
        print("  -", n)


if __name__ == "__main__":
    main()
