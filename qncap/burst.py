"""非平稳/突发到达修正。

解析 MVA 的前提是平稳马尔可夫到达, 报税截止前的真实流量既有
日间趋势又有秒级突发, 平稳模型会**系统性偏乐观**。两层修正:

1. **变异系数修正 (Allen-Cunneen G/G/m)**::

       W_GGm ≈ W_MMm * (Ca^2 + Cs^2) / 2

   Ca 为到达间隔变异系数, Cs 为服务时间变异系数;
2. **峰值等效负载**: 非平稳到达在排队系统里的等效负载按
   峰值区间(通常取 5 分钟)平均到达率而非全天均值计算,
   peak_factor = 峰值桶均值 / 全天均值。

模块既给出修正后的端到端预测, 也给出乐观偏差有多大,
并对峰值利用率超过 1 的情况明确给出不稳定告警。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List

import numpy as np

from .network import Network
from .mva import ModelResult, solve_approx, mm_m_wait, _demand_arrays


@dataclass
class BurstProfile:
    """一个开放类的流量统计特征。"""
    class_name: str
    arrival_cv: float                  # 到达间隔变异系数 Ca
    peak_factor: float                 # 峰值桶均值 / 全天均值
    peak_rate: float                   # 峰值等效到达率
    mean_rate: float
    service_cv: float = 1.0
    bucket_seconds: float = 300.0


@dataclass
class BurstResult:
    stationary: ModelResult
    corrected: ModelResult
    profiles: Dict[str, BurstProfile]
    optimism_gap: Dict[str, float] = field(default_factory=dict)
    peak_utilization: Dict[str, float] = field(default_factory=dict)
    unstable_stations: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


def profile_from_arrival_times(class_name: str,
                               arrival_times: np.ndarray,
                               horizon: float,
                               bucket_seconds: float = 300.0,
                               service_cv: float = 1.0) -> BurstProfile:
    """由到达时间戳直接估计突发特征。"""
    t = np.sort(np.asarray(arrival_times, dtype=float))
    if t.size < 4:
        rate = t.size / horizon
        return BurstProfile(class_name, 1.0, 1.0, rate, rate, service_cv,
                            bucket_seconds)
    intervals = np.diff(t)
    ca = float(intervals.std(ddof=1) / intervals.mean())
    edges = np.arange(0, horizon + bucket_seconds, bucket_seconds)
    counts, _ = np.histogram(t, bins=edges)
    bucket_rates = counts / bucket_seconds
    mean_rate = t.size / horizon
    # 峰值: 取最高 5% 桶的均值(至少一个桶), 避免单点抖动
    k = max(1, int(math.ceil(0.05 * len(bucket_rates))))
    peak_rate = float(np.mean(np.sort(bucket_rates)[-k:]))
    return BurstProfile(class_name, max(ca, 0.0),
                        peak_rate / max(mean_rate, 1e-12),
                        peak_rate, mean_rate, service_cv, bucket_seconds)


def profile_from_rate_function(class_name: str,
                               rate_fn: Callable[[float], float],
                               horizon: float,
                               bucket_seconds: float = 300.0,
                               service_cv: float = 1.0) -> BurstProfile:
    """由非齐次到达率函数 lambda(t) 估计突发特征(数值积分)。"""
    edges = np.arange(0, horizon + bucket_seconds, bucket_seconds)
    mid = (edges[:-1] + edges[1:]) / 2
    rates = np.array([rate_fn(x) for x in mid])
    mean_rate = float(rates.mean())
    k = max(1, int(math.ceil(0.05 * len(rates))))
    peak_rate = float(np.mean(np.sort(rates)[-k:]))
    # 非齐次泊松的间隔 CV 解析难求, 用强度变异的经验折算 (>=1)
    ca = float(math.sqrt(1.0 + (rates.std() / max(mean_rate, 1e-12)) ** 2))
    return BurstProfile(class_name, ca,
                        peak_rate / max(mean_rate, 1e-12),
                        peak_rate, mean_rate, service_cv, bucket_seconds)


def gg_m_wait(m: int, rate: float, s_eff: float,
              ca: float, cs: float) -> float:
    """Allen-Cunneen G/G/m 平均等待时间。"""
    w_mm = mm_m_wait(m, rate, s_eff)
    if not math.isfinite(w_mm):
        return math.inf
    return w_mm * (ca * ca + cs * cs) / 2.0


def solve_with_burst(net: Network,
                     profiles: Dict[str, BurstProfile],
                     tol: float = 1e-7,
                     max_iter: int = 500) -> BurstResult:
    """给出平稳预测与突发修正预测两份结果。

    修正做法: 开放类到达率替换为峰值等效率, 且在站点等待项上
    乘 Allen-Cunneen 变异系数; 闭合类不变(其突发由人口增长表达)。
    """
    stationary = solve_approx(net, tol=tol, max_iter=max_iter)

    # 峰值开放到达率下的不动点
    peak_rates = {c: p.peak_rate for c, p in profiles.items()}
    corrected = solve_approx(net, tol=tol, max_iter=max_iter,
                             open_rates=peak_rates)

    # Allen-Cunneen 二次修正: 对开放类的站点等待部分乘变异因子
    snames, cnames, D, V = _demand_arrays(net)
    open_idxs = [i for i, c in enumerate(cnames)
                 if net.classes[c].kind == "open"]
    for i in open_idxs:
        p = profiles.get(cnames[i])
        if p is None:
            continue
        factor = (p.arrival_cv ** 2 + p.service_cv ** 2) / 2.0
        for j, s in enumerate(snames):
            st = net.stations[s]
            if st.sched == "is" or V[i, j] <= 0:
                continue
            r = corrected.per_class_residence[cnames[i]][s]
            own_svc = D[i, j]
            wait = max(r - own_svc, 0.0)
            corrected.per_class_residence[cnames[i]][s] = (
                own_svc + wait * factor)
        corrected.class_response[cnames[i]] = sum(
            corrected.per_class_residence[cnames[i]].values())

    # 峰值利用率与稳定性
    peak_util: Dict[str, float] = {}
    unstable: List[str] = []
    for j, s in enumerate(snames):
        st = net.stations[s]
        if st.sched == "is":
            peak_util[s] = corrected.station_utilization[s]
            continue
        work = 0.0
        for i, c in enumerate(cnames):
            x = (peak_rates[c] if c in peak_rates
                 else corrected.class_throughput[c])
            work += x * D[i, j]
        u = work / st.servers
        peak_util[s] = min(u, 1.0)
        if u >= 0.98:
            unstable.append(s)
    corrected.station_utilization.update(peak_util)
    corrected.method = "burst-corrected MVA"

    gap = {
        c: (corrected.class_response[c]
            - stationary.class_response[c])
        for c in cnames}
    notes: List[str] = []
    if unstable:
        notes.append(
            "峰值负载下站点 " + ", ".join(unstable)
            + " 利用率>=98%, 排队趋于发散, 预测值为下界; "
              "必须扩容或削峰(去年的故障形态)")
    for c, p in profiles.items():
        if p.arrival_cv > 1.5 or p.peak_factor > 2.0:
            notes.append(
                f"类 {c}: 到达间隔 CV={p.arrival_cv:.2f}, "
                f"峰值因子={p.peak_factor:.2f}, 平稳模型显著偏乐观")
    return BurstResult(stationary=stationary, corrected=corrected,
                       profiles=profiles, optimism_gap=gap,
                       peak_utilization=peak_util,
                       unstable_stations=unstable, notes=notes)
