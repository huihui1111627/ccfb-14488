# -*- coding: utf-8 -*-
"""容量规划问答：用户数翻倍、瓶颈定位与转移、最少台数规划。

统一通过分层求解器 :class:`captool.layered.LayeredSolver` 求解（无资源池
时退化为普通 MVA），保证“连接池先饱和、扩容应用也无效”这类嵌套瓶颈
能被正确识别——这正是去年“加了一倍应用服务器仍宕机”的失败模式。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from .network import QueueNetwork, IS
from .layered import LayeredSolver, LayeredSolution


# ---------------------------------------------------------------------- #
# 统一求解入口
# ---------------------------------------------------------------------- #
def _solve(net: QueueNetwork,
           cv: Optional[Dict[str, tuple]] = None) -> LayeredSolution:
    return LayeredSolver(net).solve(cv=cv)


def _limiting_rows(sol: LayeredSolution) -> List[dict]:
    """把可饱和的排队站点（排除 IS 纯延迟）与资源池按利用率降序返回。

    IS 站有无限服务台，其“利用率”实为 erlang 负荷（可超过 1），不构成
    容量瓶颈，必须从瓶颈候选中剔除，否则会掩盖真正的排队资源。
    """
    rows = [r for r in sol.station_view()
            if not (r["kind"] == "station" and r["discipline"] == IS)]
    return sorted(rows, key=lambda r: r["util"], reverse=True)


def bottleneck(sol: LayeredSolution) -> dict:
    """当前系统的首要瓶颈（站点或资源池）。"""
    return _limiting_rows(sol)[0]


# ---------------------------------------------------------------------- #
# 1. 用户数/到达率变化的 What-if
# ---------------------------------------------------------------------- #
@dataclass
class WhatIfResult:
    baseline: LayeredSolution
    scaled: LayeredSolution
    population_scale: float
    arrival_scale: float
    rt_ratio: np.ndarray                 # 各类响应时间变化倍数
    new_bottleneck: dict
    stable: bool

    def summary(self) -> str:
        net = self.baseline.base.net
        lines = [f"=== What-if：闭合用户 ×{self.population_scale}，"
                 f"开放到达 ×{self.arrival_scale} ==="]
        lines.append(f"{'类':<10}{'原RT(s)':>10}{'新RT(s)':>10}"
                     f"{'倍数':>8}{'原吞吐':>10}{'新吞吐':>10}")
        for ci, c in enumerate(net.classes):
            lines.append(
                f"{c.name:<10}{self.baseline.R[ci]:10.3f}"
                f"{self.scaled.R[ci]:10.3f}{self.rt_ratio[ci]:8.2f}"
                f"{self.baseline.X[ci]:10.3f}{self.scaled.X[ci]:10.3f}")
        lines.append("扩容后首要瓶颈: {name}（{kind}，利用率 {util:.1%}，"
                     "{servers} 台/令牌）".format(**self.new_bottleneck))
        if not self.stable:
            lines.append("警告：扩容后系统不稳定（存在饱和资源），"
                         "响应时间预测为过载外推")
        return "\n".join(lines)


def what_if(net: QueueNetwork,
            population_scale: float = 2.0,
            arrival_scale: float = 2.0,
            cv: Optional[Dict[str, tuple]] = None) -> WhatIfResult:
    """预测用户数/到达率按比例增长后的响应时间、吞吐与瓶颈。"""
    base_sol = _solve(net, cv=cv)
    scaled_net = net.copy(population_scale=population_scale,
                          arrival_scale=arrival_scale)
    scaled_sol = _solve(scaled_net, cv=cv)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(base_sol.R > 0,
                         scaled_sol.R / np.maximum(base_sol.R, 1e-12),
                         np.nan)
    return WhatIfResult(base_sol, scaled_sol, population_scale,
                        arrival_scale, ratio,
                        bottleneck(scaled_sol), scaled_sol.stable)


# ---------------------------------------------------------------------- #
# 2. 瓶颈转移链
# ---------------------------------------------------------------------- #
@dataclass
class BottleneckStep:
    order: int
    name: str
    kind: str
    servers: int
    util: float
    action: str                          # 为解除该瓶颈采取的扩容动作
    after_solution: LayeredSolution


@dataclass
class BottleneckChain:
    chain: List[BottleneckStep] = field(default_factory=list)

    def names(self) -> List[str]:
        return [s.name for s in self.chain]

    def summary(self) -> str:
        lines = ["=== 瓶颈转移链（每解除一个，下一个自动浮现）==="]
        for st in self.chain:
            lines.append(
                f"{st.order}. {st.name}（{st.kind}，{st.servers} 台/令牌，"
                f"利用率 {st.util:.1%}）→ {st.action}")
        return "\n".join(lines)


def bottleneck_chain(net: QueueNetwork,
                     max_steps: int = 6,
                     util_headroom: float = 0.75,
                     cv: Optional[Dict[str, tuple]] = None
                     ) -> BottleneckChain:
    """依次定位并解除瓶颈，给出瓶颈转移顺序。

    解除动作：给当前瓶颈增加 50%（向上取整，至少 1）服务台/令牌，
    然后重解；当所有资源利用率低于 ``util_headroom`` 或链条达到
    ``max_steps`` 时停止。资源池与普通站点同等参与排序，因此可以
    正确预测“先扩连接池，再扩数据库，再扩应用”的真实顺序。
    """
    chain = BottleneckChain()
    work = net
    for order in range(1, max_steps + 1):
        sol = _solve(work, cv=cv)
        rows = _limiting_rows(sol)
        top = rows[0]
        if top["util"] < util_headroom:
            break
        # 计算扩容动作
        if top["kind"] == "pool":
            cur = next(p.capacity for p in work.pools if p.name == top["name"])
            add = max(1, int(np.ceil(cur * 0.5)))
            action = f"资源池 {top['name']} 容量 {cur} → {cur + add}"
            work = work.copy(pool_override={top["name"]: cur + add})
        else:
            st = work.stations[work.si(top["name"])]
            cur = st.servers
            add = max(1, int(np.ceil(cur * 0.5)))
            action = f"站点 {top['name']} 服务台 {cur} → {cur + add}"
            work = work.copy(servers_override={top["name"]: cur + add})
        after = _solve(work, cv=cv)
        chain.chain.append(BottleneckStep(
            order, top["name"], top["kind"], top["servers"],
            float(top["util"]), action, after))
    return chain


# ---------------------------------------------------------------------- #
# 3. 达到响应时间目标的最少台数
# ---------------------------------------------------------------------- #
@dataclass
class SizingResult:
    target_rt: Dict[str, float]
    required_servers: Dict[str, int]     # 站点/池名 -> 最少台数（含池）
    final_solution: LayeredSolution
    achieved_rt: Dict[str, float]
    feasible: bool
    steps_taken: int

    def summary(self) -> str:
        lines = ["=== 最少台数规划（贪心扩容直到满足响应时间目标）==="]
        lines.append("扩容方案: " + ", ".join(
            f"{name}={n}" for name, n in self.required_servers.items()))
        lines.append(f"{'类':<10}{'目标RT(s)':>11}{'达成RT(s)':>11}{'是否达标':>10}")
        for cname, tgt in self.target_rt.items():
            got = self.achieved_rt.get(cname, np.nan)
            lines.append(f"{cname:<10}{tgt:11.3f}{got:11.3f}"
                         f"{'是' if got <= tgt else '否':>10}")
        if not self.feasible:
            lines.append("警告：在扩容上限内未能满足全部目标，"
                         "请检查需求是否超出单站点物理上限")
        return "\n".join(lines)


def min_servers_for_target(net: QueueNetwork,
                           target_rt: Dict[str, float],
                           max_add_per_resource: int = 64,
                           cv: Optional[Dict[str, tuple]] = None
                           ) -> SizingResult:
    """贪心搜索每个站点/资源池的最少服务台数，使各类 RT 达标。

    每一步在当前解里选择“单位增量带来的受限类 RT 改善最大”的资源
    扩容 1 台；同效时优先当前利用率最高者（瓶颈优先，避免浪费）。
    资源池令牌与站点服务台统一参与搜索。
    """
    target_idx = {net.ci(name): t for name, t in target_rt.items()}
    # 物理可行性预检：即使零排队，每类 RT 也不低于其裸服务时间之和。
    # 目标低于该下限时扩容永远无法满足，直接判定不可行并指出缺口。
    bare = np.sum(net.demands, axis=1)
    infeasible = {net.classes[ci].name: float(bare[ci])
                  for ci, t in target_idx.items() if t < bare[ci] - 1e-9}
    if infeasible:
        base = _solve(net, cv=cv)
        return SizingResult(
            target_rt=target_rt,
            required_servers={**{s.name: s.servers for s in net.stations
                                 if s.discipline != IS},
                              **{p.name: p.capacity for p in net.pools}},
            final_solution=base,
            achieved_rt={net.classes[ci].name: float(base.R[ci])
                         for ci in target_idx},
            feasible=False, steps_taken=0)
    servers = {s.name: s.servers for s in net.stations
               if s.discipline != IS}
    pools = {p.name: p.capacity for p in net.pools}
    added = {**{k: 0 for k in servers}, **{k: 0 for k in pools}}
    work = net
    steps = 0

    def meets(sol: LayeredSolution) -> bool:
        return all(sol.R[ci] <= t for ci, t in target_idx.items()) \
            and sol.stable

    sol = _solve(work, cv=cv)
    while not meets(sol):
        best = None           # (改善量, 利用率, 资源名, 类型)
        candidates = list(servers) + list(pools)
        for name in candidates:
            if added[name] >= max_add_per_resource:
                continue
            if name in servers:
                trial = work.copy(
                    servers_override={name: servers[name] + 1})
            else:
                trial = work.copy(
                    pool_override={name: pools[name] + 1})
            trial_sol = _solve(trial, cv=cv)
            # 受限最严重类别的 RT 改善（越大越优先）
            improvement = 0.0
            for ci, t in target_idx.items():
                gap_before = max(0.0, sol.R[ci] - t)
                gap_after = max(0.0, trial_sol.R[ci] - t)
                improvement += gap_before - gap_after
            util_here = next((r["util"] for r in trial_sol.station_view()
                              if r["name"] == name), 0.0)
            key = (improvement, util_here)
            if best is None or key > best[0]:
                best = (key, name, trial, trial_sol)
        if best is None or best[0][0] <= 1e-9:
            return SizingResult(target_rt, {**servers, **pools}, sol,
                                {net.classes[ci].name: float(sol.R[ci])
                                 for ci in target_idx},
                                feasible=False, steps_taken=steps)
        _, name, work, sol = best
        if name in servers:
            servers[name] += 1
        else:
            pools[name] += 1
        added[name] += 1
        steps += 1

    achieved = {net.classes[ci].name: float(sol.R[ci])
                for ci in target_idx}
    return SizingResult(target_rt, {**servers, **pools}, sol, achieved,
                        feasible=True, steps_taken=steps)
