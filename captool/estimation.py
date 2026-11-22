# -*- coding: utf-8 -*-
"""服务需求反推：从监控数据估计 D[c,s]。

原理
====
服务定律（Service Law）：站点 s 的利用率等于所有类在该站的吞吐-需求积之和

    U[t,s] = Σ_c X[t,c] · D[c,s] + 噪声

其中 U[t,s] 为采样窗口 t 内站点 s 的平均利用率，X[t,c] 为类 c 在窗口 t
内的平均吞吐（个/秒）。多窗口监控数据给出超定方程组，逐站点做非负最小
二乘（NNLS，Lawson-Hanson 主动集算法，纯 NumPy 实现，不依赖 SciPy）：

    min_D ≥ 0  ‖A D - b‖²,   A[t,c]=X[t,c], b[t]=U[t,s]

闭合类的思考时间由 Little 定律反推：N[c] = X[c]·(R[c]+Z[c])，
若监控给出了端到端响应时间 R，则 Z = N/X - R。

反推结果可直接喂给 :func:`captool.mva.solve` 的 demands 参数。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from .network import QueueNetwork, CLOSED, IS


# ---------------------------------------------------------------------- #
# 纯 NumPy 实现的 Lawson-Hanson 主动集 NNLS
# ---------------------------------------------------------------------- #
def nnls(A: np.ndarray, b: np.ndarray, max_iter: Optional[int] = None
         ) -> np.ndarray:
    """非负最小二乘 min_{x>=0} ||Ax-b||₂。

    经典 Lawson-Hanson 主动集法；矩阵规模通常为 (监控窗口数 × 类数)，
    很小，直接用 :func:`numpy.linalg.lstsq` 解自由变量子问题。
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    m, n = A.shape
    max_iter = max_iter or 3 * n
    x = np.zeros(n)
    P = np.zeros(n, dtype=bool)          # 主动集（自由变量集）
    for _ in range(max_iter):
        w = A.T @ (b - A @ x)
        if not np.any(~P):
            break
        w_masked = np.where(~P, w, -np.inf)
        j = int(np.argmax(w_masked))
        if w[j] <= 1e-12:
            break
        P[j] = True
        # 在自由集上解无约束最小二乘
        idx = np.where(P)[0]
        s = np.zeros(n)
        sol, *_ = np.linalg.lstsq(A[:, idx], b, rcond=None)
        s[idx] = sol
        # 内层循环：保证可行性
        while np.any(s[idx] <= 1e-12):
            alpha = min(x[i] / (x[i] - s[i])
                        for i in idx if s[i] <= 1e-12)
            x = x + alpha * (s - x)
            for i in [i for i in idx if abs(x[i]) <= 1e-12]:
                P[i] = False
            idx = np.where(P)[0]
            if len(idx) == 0:
                s = np.zeros(n)
                break
            s = np.zeros(n)
            sol, *_ = np.linalg.lstsq(A[:, idx], b, rcond=None)
            s[idx] = sol
        x = s
    return x


@dataclass
class MonitorSample:
    """一个监控窗口（例如 5 分钟聚合）的观测值。"""
    throughput: Dict[str, float]          # 类名 -> 平均吞吐（个/秒）
    utilization: Dict[str, float]         # 站点名 -> 平均利用率（0~1）
    response: Optional[Dict[str, float]] = None   # 类名 -> 平均响应时间（秒）
    population: Optional[Dict[str, float]] = None # 闭合类名 -> 在线用户数


@dataclass
class DemandEstimate:
    demands: np.ndarray                    # D[c,s]，与网络同序
    r2: np.ndarray                         # 每个站点拟合优度
    residual: np.ndarray                   # 每个站点的 RMSE（利用率量纲）
    think: np.ndarray                      # 闭合类思考时间估计（秒）
    n_samples: int
    notes: List[str]

    def summary(self, net: QueueNetwork) -> str:
        lines = ["=== 服务需求反推结果 D[c,s]（秒）===",
                 "         " + " ".join(f"{s.name:>10}" for s in net.stations)]
        for ci, c in enumerate(net.classes):
            lines.append(f"{c.name:<9}" + " ".join(
                f"{self.demands[ci, si]:10.4f}" for si in range(net.S)))
        lines.append("拟合优度 R²: " + ", ".join(
            f"{s.name}={self.r2[si]:.3f}"
            for si, s in enumerate(net.stations)))
        lines.append("利用率 RMSE: " + ", ".join(
            f"{s.name}={self.residual[si]:.4f}"
            for si, s in enumerate(net.stations)))
        if np.any(self.think > 0):
            lines.append("思考时间: " + ", ".join(
                f"{c.name}={self.think[ci]:.2f}s"
                for ci, c in enumerate(net.classes) if self.think[ci] > 0))
        if self.notes:
            lines.append("备注: " + "; ".join(self.notes))
        return "\n".join(lines)


def infer_demands(net: QueueNetwork,
                  samples: Sequence[MonitorSample]) -> DemandEstimate:
    """由多窗口监控样本反推服务需求矩阵与闭合类思考时间。

    要求
    ----
    - 至少 1 个监控窗口；窗口数 ≥ 类数时方程超定、拟合最可靠；
    - 利用率为忙时占比（0~1），吞吐单位为个/秒，二者时间窗口口径一致；
    - 单窗口场景退化为按吞吐比例分摊的点估计，R² 不可计算（置为 NaN）。
    """
    if not samples:
        raise ValueError("至少需要一个监控窗口")
    T = len(samples)
    C, S = net.C, net.S
    A = np.zeros((T, C))
    B = np.zeros((T, S))
    for t, smp in enumerate(samples):
        for ci, c in enumerate(net.classes):
            A[t, ci] = smp.throughput.get(c.name, 0.0)
        for si, st in enumerate(net.stations):
            u = smp.utilization.get(st.name, 0.0)
            # 服务定律：忙台数（erlang 负荷）= m·U = Σ_c X[c]D[c,s]。
            # 监控给的是占比 U，需乘回服务台数；IS 无限台，U 即负荷。
            B[t, si] = u if st.discipline == IS else u * st.servers

    D = np.zeros((C, S))
    r2 = np.full(S, np.nan)
    rmse = np.zeros(S)
    notes: List[str] = []
    for si, st in enumerate(net.stations):
        b = B[:, si]
        if np.max(b) <= 1e-12:
            continue
        if T == 1:
            # 单窗口：信息不足，按各类吞吐占比做可行分摊
            total = np.sum(A[0])
            if total > 0:
                D[:, si] = A[0] / total * b[0]
            notes.append(f"站点 {st.name} 仅有单窗口观测，"
                         f"需求为按吞吐比例的点估计")
            rmse[si] = 0.0
            continue
        x = nnls(A, b)
        D[:, si] = x
        pred = A @ x
        ss_res = float(np.sum((b - pred) ** 2))
        ss_tot = float(np.sum((b - np.mean(b)) ** 2))
        r2[si] = 1.0 - ss_res / ss_tot if ss_tot > 1e-18 else np.nan
        rmse[si] = np.sqrt(ss_res / T)
        if np.isfinite(r2[si]) and r2[si] < 0.8:
            notes.append(f"站点 {st.name} 拟合优度偏低 R²={r2[si]:.2f}，"
                         f"建议增加监控窗口或检查类划分")

    # 没有任何访问的类-站对保持 0；校验反推矩阵与路由结构一致
    V = net.visits
    D = np.where(V > 0, D, 0.0)

    # 思考时间反推（取所有可用窗口的中位数，降低噪声）
    think = np.zeros(C)
    for ci, c in enumerate(net.classes):
        if c.kind != CLOSED:
            continue
        vals = []
        for smp in samples:
            x_rate = smp.throughput.get(c.name, 0.0)
            n_users = (smp.population or {}).get(c.name, c.population)
            if x_rate <= 0:
                continue
            r_obs = (smp.response or {}).get(c.name)
            if r_obs is not None:
                vals.append(n_users / x_rate - r_obs)
            else:
                # 无响应时间监控时，N/X 给出 R+Z 的上界估计
                vals.append(n_users / x_rate)
        if vals:
            think[ci] = max(0.0, float(np.median(vals)))
    if np.any(think > 0) and not any(
            smp.response for smp in samples):
        notes.append("监控未提供响应时间，思考时间为 N/X 上界（含R），"
                     "建议补充响应时间指标")
    return DemandEstimate(D, r2, rmse, think, T, notes)


def replace_network_demands(net: QueueNetwork,
                            est: DemandEstimate) -> QueueNetwork:
    """用反推需求构造等效网络：把每类在每站的总需求压成单步服务时间。

    保留原路由的池 acquire/release 结构（分层模型仍需要），
    仅替换服务时间数值。
    """
    from .network import Station, RequestClass, Step
    V = net.visits
    new_classes = []
    for ci, c in enumerate(net.classes):
        route = []
        counts: Dict[str, int] = {}
        for step in c.route:
            si = net.si(step.station)
            counts[step.station] = counts.get(step.station, 0) + 1
            v = V[ci, si]
            per_visit = est.demands[ci, si] / v if v > 0 else 0.0
            route.append(Step(step.station, float(per_visit),
                              step.acquire, step.release))
        new_classes.append(RequestClass(
            c.name, c.kind, route,
            population=c.population,
            think=est.think[ci] if est.think[ci] > 0 else c.think,
            arrival=c.arrival, desc=c.desc))
    return QueueNetwork([Station(s.name, s.servers, s.discipline, s.desc)
                         for s in net.stations], new_classes,
                        pools=list(net.pools), name=net.name)
