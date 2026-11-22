# -*- coding: utf-8 -*-
"""解析模型的仿真验收。

在 light / medium / heavy 三档负载下分别运行：
1. 分层解析模型（LayeredSolver，内部为 MVA + Erlang-C）；
2. 离散事件仿真（多种子、批均值 95% 置信区间）。

逐类比较响应时间、吞吐，逐站点比较利用率，并核对：
- 解析预测是否落入仿真置信区间；
- 相对偏差是否在工程可接受范围内（轻/中载 15%，近饱和 25%）；
- 瓶颈资源排序是否一致、瓶颈转移链预测是否正确。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .network import QueueNetwork, IS
from .layered import LayeredSolver, LayeredSolution
from .simulation import simulate_replicates, SimResult
from .scenario import LOAD_LEVELS, load_network
from . import sizing

# 工程验收阈值
ERROR_LIMIT = {"light": 0.15, "medium": 0.15, "heavy": 0.25}


@dataclass
class MetricRow:
    name: str
    analytic: float
    sim: float
    sim_ci: float
    rel_err: float
    within_ci: bool
    within_limit: bool


@dataclass
class LevelReport:
    level: str
    rows: List[MetricRow] = field(default_factory=list)
    bottleneck_analytic: str = ""
    bottleneck_sim: str = ""
    bottleneck_match: bool = False
    max_err: float = 0.0
    n_within_limit: int = 0
    n_total: int = 0
    stable: bool = True

    def summary(self) -> str:
        head = (f"--- {self.level}：首要瓶颈 解析={self.bottleneck_analytic} "
                f"仿真={self.bottleneck_sim} "
                f"{'✓' if self.bottleneck_match else '✗'}；"
                f"最大偏差 {self.max_err:.1%}，"
                f"达标 {self.n_within_limit}/{self.n_total} ---")
        lines = [head,
                 f"{'指标':<22}{'解析':>10}{'仿真':>10}{'±CI':>8}"
                 f"{'偏差':>9}{'入CI':>6}{'达标':>6}"]
        for r in self.rows:
            lines.append(
                f"{r.name:<22}{r.analytic:10.4f}{r.sim:10.4f}"
                f"{r.sim_ci:8.4f}{r.rel_err:8.1%}"
                f"{'是' if r.within_ci else '否':>6}"
                f"{'是' if r.within_limit else '否':>6}")
        return "\n".join(lines)


@dataclass
class ValidationReport:
    levels: Dict[str, LevelReport]
    chain_analytic: List[str]
    chain_sim: List[str]
    chain_match: bool

    def summary(self) -> str:
        parts = [lv.summary() for lv in
                 (self.levels[k] for k in LOAD_LEVELS if k in self.levels)]
        parts.append("")
        parts.append("瓶颈转移链（解析）: " + " → ".join(self.chain_analytic))
        parts.append("瓶颈转移链（仿真）: " + " → ".join(self.chain_sim))
        parts.append("瓶颈顺序预测: "
                     + ("✓ 正确" if self.chain_match else "✗ 不一致"))
        return "\n".join(parts)

    def to_records(self) -> List[dict]:
        out = []
        for level, lv in self.levels.items():
            for r in lv.rows:
                out.append({"level": level, "metric": r.name,
                            "analytic": r.analytic, "sim": r.sim,
                            "sim_ci": r.sim_ci, "rel_err": r.rel_err,
                            "within_ci": r.within_ci,
                            "within_limit": r.within_limit})
        return out


def _row(name: str, analytic: float, sim: float, ci: float,
         limit: float, abs_tol: float = 0.02) -> MetricRow:
    """对比单行指标。

    相对误差用于有量级的指标；当仿真值本身接近 0（如轻载下的池等待），
    相对误差会因除零放大为假象，此时改用绝对误差并以 ``abs_tol`` 判定。
    """
    diff = abs(analytic - sim)
    if abs(sim) >= abs_tol:
        err = diff / abs(sim)
        within_ci = diff <= max(ci, 0.02 * abs(sim))
        within_limit = err <= limit
    else:
        # 仿真值近零：相对误差无意义，报告归一化绝对误差（除以容差），
        # 数值落在 0~1 即达标，避免除零产生的上千百分比假象。
        err = diff / abs_tol
        within_ci = diff <= max(ci, abs_tol)
        within_limit = diff <= abs_tol
    return MetricRow(name, float(analytic), float(sim), float(ci),
                     float(err), within_ci, within_limit)


def validate_level(net: QueueNetwork, level: str,
                   sim_kwargs: Optional[dict] = None
                   ) -> (LevelReport, LayeredSolution, SimResult):
    """对单个负载档位执行解析 vs 仿真对比。"""
    sim_kwargs = sim_kwargs or {}
    analytic = LayeredSolver(net).solve()
    sim = simulate_replicates(net, **sim_kwargs)
    limit = ERROR_LIMIT[level]
    rep = LevelReport(level=level, stable=analytic.stable)

    for ci, c in enumerate(net.classes):
        rep.rows.append(_row(f"RT[{c.name}](s)", analytic.R[ci],
                             sim.R[ci], sim.R_ci[ci], limit))
        rep.rows.append(_row(f"X[{c.name}](/s)", analytic.X[ci],
                             sim.X[ci], sim.X_ci[ci], limit))
    for si, st in enumerate(net.stations):
        label = "负荷" if st.discipline == IS else "利用率"
        rep.rows.append(_row(f"{label}[{st.name}]",
                             analytic.util[si], sim.util[si],
                             sim.util_ci[si], limit))
    for k, pname in enumerate(sim.pool_names):
        ai = list(analytic.pool_names).index(pname)
        rep.rows.append(_row(f"池利用率[{pname}]",
                             analytic.pool_util[ai], sim.pool_util[k],
                             0.0, limit))
        rep.rows.append(_row(f"池等待[{pname}](s)",
                             analytic.pool_wait[ai], sim.pool_wait[k],
                             0.0, limit))

    rep.max_err = max(r.rel_err for r in rep.rows)
    rep.n_total = len(rep.rows)
    rep.n_within_limit = sum(r.within_limit for r in rep.rows)
    rep.bottleneck_analytic = sizing.bottleneck(analytic)["name"]
    # 仿真瓶颈：站点与池统一按利用率排序
    sim_view = _sim_bottleneck(net, sim)
    rep.bottleneck_sim = sim_view
    rep.bottleneck_match = rep.bottleneck_analytic == rep.bottleneck_sim
    return rep, analytic, sim


def _sim_bottleneck(net: QueueNetwork, sim: SimResult) -> str:
    best_name, best_util = None, -1.0
    for si, st in enumerate(net.stations):
        u = sim.util[si] if st.discipline != IS else -1.0
        if u > best_util:
            best_name, best_util = st.name, u
    for k, pname in enumerate(sim.pool_names):
        if sim.pool_util[k] > best_util:
            best_name, best_util = pname, sim.pool_util[k]
    return best_name


def _sim_chain(net: QueueNetwork, max_steps: int = 5) -> List[str]:
    """仿真口径的瓶颈转移链：每轮取当前瓶颈、扩容 50% 后重跑仿真。"""
    chain: List[str] = []
    work = net
    servers = {s.name: s.servers for s in net.stations
               if s.discipline != IS}
    pools = {p.name: p.capacity for p in net.pools}
    for _ in range(max_steps):
        sim = simulate_replicates(
            work, n_rep=2,
            max_seconds=180.0, min_completed=20000)
        name = _sim_bottleneck(work, sim)
        if name in chain:
            break
        chain.append(name)
        if name in pools:
            cur = pools[name]
            pools[name] = cur + max(1, int(np.ceil(cur * 0.5)))
            work = work.copy(pool_override={name: pools[name]})
        else:
            cur = servers[name]
            servers[name] = cur + max(1, int(np.ceil(cur * 0.5)))
            work = work.copy(servers_override={name: servers[name]})
    return chain


def run_validation(sim_seconds: float = 300.0,
                   n_rep: int = 3,
                   min_completed: int = 30000) -> ValidationReport:
    """完整三档验收 + 瓶颈转移链核对（命令行与页面共用）。"""
    sim_kwargs = dict(n_rep=n_rep, max_seconds=sim_seconds,
                      min_completed=min_completed)
    reports: Dict[str, LevelReport] = {}
    heavy_net = None
    for level in LOAD_LEVELS:
        net = load_network(level)
        if level == "heavy":
            heavy_net = net
        rep, _, _ = validate_level(net, level, sim_kwargs=sim_kwargs)
        reports[level] = rep

    # 瓶颈转移链在近饱和场景上推演（headroom 取低值以展开完整链条）
    chain_a = sizing.bottleneck_chain(
        heavy_net, max_steps=5, util_headroom=0.4).names()
    chain_s = _sim_chain(heavy_net, max_steps=5)
    # 顺序核对：两条链的共同瓶颈须保持相同的相对顺序（允许扩容步长不同
    # 导致的个别条目差异），首要瓶颈必须一致。
    common = [n for n in chain_a if n in chain_s]
    common_s = [n for n in chain_s if n in chain_a]
    match = (bool(chain_a) and bool(chain_s)
             and chain_a[0] == chain_s[0] and common == common_s)
    return ValidationReport(reports, chain_a, chain_s, match)
