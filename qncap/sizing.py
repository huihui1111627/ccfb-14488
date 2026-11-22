"""容量规划: 瓶颈识别、瓶颈转移、最小台数规划。

三个核心问题:

* :func:`rank_bottlenecks` —— 当前配置下站点瓶颈排序。
  排序依据是"相对压力" U_i/m_i 的归一化饱和余量, 同时结合
  队列驻留时间, 池饱和单独标注;
* :func:`bottleneck_shift_sequence` —— 逐次解除当前瓶颈
  (加服务台/扩池), 重新求解, 给出瓶颈转移顺序,
  即"数据库解了以后下一个卡住的是谁";
* :func:`min_servers_for_target` —— 给定端到端响应时间目标,
  求每个站点的最少服务台数。贪心 + 局部边界搜索,
  并校验目标在渐近容量下是否可达(不可达直接说明)。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .network import Network
from .mva import ModelResult


# 求解器可注入, 分层网络传 layered.solve_layered
Solver = Callable[..., ModelResult]


@dataclass
class BottleneckRank:
    station: str
    utilization: float
    pressure: float                    # 归一化压力 0~1
    residence: float
    queue_length: float
    pool_saturated: bool
    reason: str


@dataclass
class ShiftStep:
    step: int
    bottleneck: str
    action: str
    utilization_before: float
    utilization_after: float
    new_bottleneck: Optional[str]
    response_times: Dict[str, float]


def rank_bottlenecks(net: Network, result: ModelResult,
                     pool_info: Optional[Dict[str, Dict[str, float]]] = None
                     ) -> List[BottleneckRank]:
    """按饱和压力给站点排序(压力最大者即当前瓶颈)。"""
    ranks: List[BottleneckRank] = []
    max_u = max(result.station_utilization.values()) or 1.0
    for s in net.stations:
        st = net.stations[s]
        u = result.station_utilization[s]
        pressure = u if st.sched != "is" else u / (u + 1.0)
        pool_sat = False
        reason_parts = [f"利用率 {u:.1%}"]
        if pool_info and s in pool_info:
            p = pool_info[s]["saturation_prob"]
            pool_sat = p > 0.3
            reason_parts.append(f"池饱和概率 {p:.1%}")
        if u >= 0.9 * max_u and u > 0.5:
            reason_parts.append("接近饱和, 排队等待快速上升")
        ranks.append(BottleneckRank(
            station=s, utilization=u, pressure=pressure,
            residence=result.station_residence.get(s, 0.0),
            queue_length=result.station_queue_length.get(s, 0.0),
            pool_saturated=pool_sat,
            reason="; ".join(reason_parts)))
    ranks.sort(key=lambda r: (r.pool_saturated, r.pressure), reverse=True)
    return ranks


def _set_capacity(net: Network, station: str, servers: Optional[int] = None,
                  pool: Optional[int] = None) -> None:
    st = net.stations[station]
    if servers is not None and st.sched != "is":
        st.servers = servers
    if pool is not None and st.pool_capacity is not None:
        st.pool_capacity = pool


def bottleneck_shift_sequence(net: Network, solver: Solver,
                              max_steps: int = 6,
                              solver_kwargs: Optional[dict] = None
                              ) -> List[ShiftStep]:
    """逐次给当前瓶颈加一台(池则 +25% 向上取整), 观察瓶颈转移。"""
    solver_kwargs = solver_kwargs or {}
    seq: List[ShiftStep] = []
    current = net.copy()
    for step in range(1, max_steps + 1):
        res = solver(current, **solver_kwargs)
        ranks = rank_bottlenecks(current, res)
        # 选非 IS、利用率最高的可扩容站点
        cand = [r for r in ranks
                if current.stations[r.station].sched != "is"]
        if not cand:
            break
        top = cand[0]
        st = current.stations[top.station]
        old_u = top.utilization
        if st.pool_capacity is not None and top.pool_saturated:
            new_pool = max(st.pool_capacity + 1,
                           int(np.ceil(st.pool_capacity * 1.25)))
            action = f"连接/线程池 {st.pool_capacity} -> {new_pool}"
            _set_capacity(current, top.station, pool=new_pool)
        else:
            new_m = (st.servers or 1) + 1
            action = f"服务台 {st.servers} -> {new_m}"
            _set_capacity(current, top.station, servers=new_m)
            if st.pool_capacity is not None and st.pool_capacity < new_m:
                # 服务台增加而池更小没有意义, 同步扩池
                st.pool_capacity = max(st.pool_capacity, new_m)
        res2 = solver(current, **solver_kwargs)
        ranks2 = rank_bottlenecks(current, res2)
        next_b = next((r.station for r in ranks2
                       if current.stations[r.station].sched != "is"), None)
        seq.append(ShiftStep(
            step=step, bottleneck=top.station, action=action,
            utilization_before=old_u,
            utilization_after=res2.station_utilization[top.station],
            new_bottleneck=next_b,
            response_times=dict(res2.class_response)))
        if next_b == top.station and res2.station_utilization[top.station] < 0.5:
            break
    return seq


def min_servers_for_target(net: Network, solver: Solver,
                           target_response: Dict[str, float],
                           max_servers: int = 128,
                           solver_kwargs: Optional[dict] = None
                           ) -> Tuple[Dict[str, int], ModelResult, List[str]]:
    """求满足各类端到端响应时间目标的最小服务台配置。

    先用 :func:`asymptotic_ceiling` 校验目标可达性(无限台数下的响应
    下限); 不可达的类直接说明。可达时贪心: 每轮给超标类路径上
    排队贡献最大的站点 +1 台, 直到全部达标或触上限。
    """
    solver_kwargs = solver_kwargs or {}
    work = net.copy()
    notes: List[str] = []
    ceiling = asymptotic_ceiling(net)
    unreachable = [c for c, t in target_response.items()
                   if ceiling.get(c, math.inf) >= t]
    if unreachable:
        notes.append(
            "以下类的目标低于无限扩容的响应下限(纯服务+纯延迟), "
            "加机器不可达, 需优化代码/减少调用: "
            + ", ".join(f"{c}(下限{ceiling[c]:.3f}s/目标"
                        f"{target_response[c]:.3f}s)"
                        for c in unreachable))
        # 把不可达类目标放宽到下限的 1.05 倍, 继续求其余类的配置
        target_response = dict(target_response)
        for c in unreachable:
            target_response[c] = ceiling[c] * 1.05

    for _ in range(max_servers * len(work.stations)):
        res = solver(work, **solver_kwargs)
        violated = [c for c, t in target_response.items()
                    if res.class_response.get(c, math.inf) > t]
        if not violated:
            break
        # 超标类路径上, 选"排队驻留时间"最大的可扩容站点(而非利用率,
        # 因为低利用率站点也可能因多次访问累积驻留)
        candidates: Dict[str, float] = {}
        for c in violated:
            for s, residence in res.per_class_residence[c].items():
                st = work.stations[s]
                if st.sched == "is":
                    continue
                # 排队部分 = 驻留 - 自身服务需求
                own = net.per_class_demands()[c].get(s, 0.0)
                queue_part = max(residence - own, 0.0)
                candidates[s] = max(candidates.get(s, 0.0), queue_part)
        if not candidates:
            notes.append("剩余超标类路径上没有可扩容站点, 目标不可达")
            break
        target_station = max(candidates, key=candidates.get)
        st = work.stations[target_station]
        if (st.servers or 1) >= max_servers:
            notes.append(f"站点 {target_station} 触及 {max_servers} 台上限")
            break
        _set_capacity(work, target_station, servers=(st.servers or 1) + 1)
        if st.pool_capacity is not None:
            st.pool_capacity = max(st.pool_capacity, st.servers)
    else:
        notes.append("搜索轮次耗尽")

    final = solver(work, **solver_kwargs)
    for c, t in target_response.items():
        got = final.class_response[c]
        status = "达标" if got <= t else "未达标"
        notes.append(f"类 {c}: 目标 {t:.3f}s, 预测 {got:.3f}s [{status}]")
    config = {s: (work.stations[s].servers or 0)
              for s in work.stations if work.stations[s].sched != "is"}
    return config, final, notes


def asymptotic_ceiling(net: Network) -> Dict[str, float]:
    """各类在无限服务台下的极限响应时间(只剩自身服务需求与 IS 延迟)。

    目标低于该值时, 加机器永远达不到, 必须优化代码/减少调用次数。
    """
    dem = net.per_class_demands()
    out = {}
    for cname in net.classes:
        floor = sum(d for s, d in dem[cname].items()
                    if net.stations[s].sched == "is")
        svc = sum(d for s, d in dem[cname].items()
                  if net.stations[s].sched != "is")
        out[cname] = floor + svc
    return out


def what_if_population(net: Network, solver: Solver,
                       multipliers: Dict[str, float],
                       solver_kwargs: Optional[dict] = None
                       ) -> ModelResult:
    """闭合类用户数按倍数变化后的预测(开放类不变)。"""
    solver_kwargs = solver_kwargs or {}
    work = net.copy()
    for c in work.classes.values():
        if c.kind == "closed" and c.name in multipliers:
            c.population = max(1, int(round(
                c.population * multipliers[c.name])))
    return solver(work, **solver_kwargs)
