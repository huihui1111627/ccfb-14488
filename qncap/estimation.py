"""服务需求反推: 从监控利用率/吞吐求 D[k,i]。

真实系统拿不到"一次请求在某站点的 CPU/IO 服务时间", 但监控能给:

* 各站点时间桶利用率 U_i(t)(服务台口径, 0~1);
* 各类请求的完成吞吐 X_k(t)(每秒事务数);
* 各站点服务台数 m_i。

由利用率定律::

    U_i(t) * m_i = sum_k X_k(t) * D[k,i] + 噪声

对每个站点 i, 这是关于 D[:,i] 的非负线性最小二乘问题,
本模块实现经典 active-set NNLS (Lawson-Hanson), 仅依赖 NumPy。

闭合类额外给 think time 的反推: 由响应时间 R_k 与人口 N_k,
``Z_k = N_k / X_k - R_k`` (Little 定律的闭合系统形式)。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------- #
# Active-set NNLS (Lawson & Hanson, 1974)
# ---------------------------------------------------------------------- #
def nnls(A: np.ndarray, b: np.ndarray, max_iter: Optional[int] = None,
         tol: float = 1e-10) -> Tuple[np.ndarray, float]:
    """非负最小二乘 min_{x>=0} ||Ax - b||_2。

    返回 (x, 残差范数)。纯 NumPy 实现, 用 QR (np.linalg.qr) 解子问题。
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    m, n = A.shape
    x = np.zeros(n)
    if max_iter is None:
        max_iter = 3 * n
    passive = np.zeros(n, dtype=bool)
    resid = b.copy()

    for _ in range(max_iter):
        # 负梯度方向上最大的正分量进基
        w = A.T @ resid
        if passive.all():
            break
        wmax = np.max(w[~passive])
        if wmax <= tol * (np.max(np.abs(w)) + 1.0):
            break
        j = int(np.argmax(np.where(~passive, w, -np.inf)))
        passive[j] = True
        # 在被动集上解无约束最小二乘
        while True:
            idx = np.where(passive)[0]
            Ap = A[:, idx]
            coef, *_ = np.linalg.lstsq(Ap, b, rcond=None)
            if np.all(coef > tol):
                x[:] = 0.0
                x[idx] = coef
                break
            # 有分量越界为负: 沿原解方向插值后剔除
            ratio = min(x[idx][coef <= tol]
                        / (x[idx][coef <= tol]
                           - coef[coef <= tol] + 1e-300))
            x = x + ratio * (np.where(passive,
                                      _full_solution(A, b, passive, n),
                                      0.0) - x)
            drop = idx[np.where((x[idx] <= tol) & (coef <= tol))[0]]
            passive[drop] = False
            x[drop] = 0.0
            idx2 = np.where(passive)[0]
            if idx2.size == 0:
                break
            coef2, *_ = np.linalg.lstsq(A[:, idx2], b, rcond=None)
            x[:] = 0.0
            x[idx2] = coef2
        resid = b - A @ x
    return x, float(np.linalg.norm(resid))


def _full_solution(A: np.ndarray, b: np.ndarray,
                   passive: np.ndarray, n: int) -> np.ndarray:
    idx = np.where(passive)[0]
    out = np.zeros(n)
    if idx.size:
        coef, *_ = np.linalg.lstsq(A[:, idx], b, rcond=None)
        out[idx] = coef
    return out


# ---------------------------------------------------------------------- #
# 监控数据 -> 服务需求
# ---------------------------------------------------------------------- #
@dataclass
class MonitoringData:
    """一段时间桶内的监控样本。

    throughput: [T 桶 × K 类] 各类完成吞吐 (tps);
    utilization: [T 桶 × S 站点] 站点利用率 (0~1, 总服务台口径);
    servers: 各站点服务台数;
    response_time: 可选 [K] 各类平均端到端响应时间(秒), 用于反推 think。
    """
    class_names: List[str]
    station_names: List[str]
    throughput: np.ndarray
    utilization: np.ndarray
    servers: np.ndarray
    response_time: Optional[np.ndarray] = None
    timestamps: Optional[np.ndarray] = None


@dataclass
class DemandEstimate:
    demands: Dict[str, Dict[str, float]]      # 类 -> 站点 -> D(秒)
    residual: Dict[str, float] = field(default_factory=dict)
    r_squared: Dict[str, float] = field(default_factory=dict)
    think_time: Dict[str, float] = field(default_factory=dict)
    identifiability: List[str] = field(default_factory=list)


def estimate_demands(data: MonitoringData,
                     populations: Optional[Dict[str, int]] = None
                     ) -> DemandEstimate:
    """逐站点 NNLS 反推服务需求矩阵 D[K,S]。

    ``populations`` 给闭合类人口, 配合 response_time 反推 think time。
    """
    X = np.asarray(data.throughput, dtype=float)   # T x K
    U = np.asarray(data.utilization, dtype=float)  # T x S
    K = len(data.class_names)
    S = len(data.station_names)
    if X.shape[1] != K or U.shape[1] != S:
        raise ValueError("监控数据维度与类/站点数不一致")

    D = np.zeros((K, S))
    residual: Dict[str, float] = {}
    r2: Dict[str, float] = {}
    ident: List[str] = []

    for j, s in enumerate(data.station_names):
        b = U[:, j] * data.servers[j]
        if np.linalg.matrix_rank(X) < K:
            ident.append(
                f"站点 {s}: 吞吐矩阵秩({np.linalg.matrix_rank(X)}) < "
                f"类数({K}), 需求不唯一, 给出的是最小范数解; "
                "建议增加负载差异更大的时间桶")
        d, res_norm = nnls(X, b)
        D[:, j] = d
        residual[s] = res_norm
        ss_tot = float(((b - b.mean()) ** 2).sum())
        r2[s] = 1.0 - res_norm ** 2 / ss_tot if ss_tot > 1e-12 else float("nan")

    out = DemandEstimate(
        demands={
            c: {s: float(D[k, j]) for j, s in enumerate(data.station_names)}
            for k, c in enumerate(data.class_names)},
        residual=residual, r_squared=r2, identifiability=ident)

    # 闭合类 think time: Z = N/X - R
    if populations and data.response_time is not None:
        for k, c in enumerate(data.class_names):
            if c in populations:
                x_mean = X[:, k].mean()
                if x_mean > 0:
                    out.think_time[c] = max(
                        populations[c] / x_mean - data.response_time[k], 0.0)
    return out


def make_monitoring_from_result(throughput: Dict[str, float],
                                demands: Dict[str, Dict[str, float]],
                                servers: Dict[str, int],
                                n_buckets: int = 40,
                                noise: float = 0.02,
                                rng: Optional[np.random.Generator] = None
                                ) -> MonitoringData:
    """用已知需求+吞吐合成监控样本(供端到端自检与演示)。

    时间桶吞吐围绕基准做 ±60% 的随机波动, 保证矩阵满秩可辨识;
    利用率由利用率定律叠加小噪声生成。
    """
    rng = rng or np.random.default_rng(7)
    cnames = list(throughput.keys())
    snames = list(servers.keys())
    K, S = len(cnames), len(snames)
    X = np.zeros((n_buckets, K))
    for k, c in enumerate(cnames):
        scale = rng.uniform(0.4, 1.6, n_buckets)
        X[:, k] = throughput[c] * scale
    U = np.zeros((n_buckets, S))
    D = np.array([[demands[c][s] for s in snames] for c in cnames])
    m = np.array([servers[s] for s in snames], dtype=float)
    for t in range(n_buckets):
        work = X[t] @ D
        U[t] = work / m
        U[t] = np.clip(U[t] * (1 + rng.normal(0, noise, S)), 0, 0.999)
    return MonitoringData(cnames, snames, X, U, m)
