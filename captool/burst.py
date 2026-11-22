# -*- coding: utf-8 -*-
"""突发到达与非平稳到达的修正。

解析模型默认假设泊松到达（间隔变异系数平方 ca²=1）、指数服务（cs²=1），
在税务截止日前的强突发场景下会系统性低估排队。本模块提供三层修正：

1. Allen-Cunneen / Whitt 修正
   对 GI/GI/m 队列，等待项乘以系数 (ca²+cs²)/2；由监控的到达间隔序列
   直接估计 ca²（服务时间 cs² 通常近似 1，也可由 APM 数据给出）。

2. 峰值区间（非平稳）修正
   非平稳到达不能用全天平均到达率。把时间轴切成短窗口，取“最不利窗口”
   的有效到达率 λ_peak（再做一次 SRS 简单随机平滑，避免单点抖动被当成
   容量需求），用峰值负荷重解网络，给出有容量保证的预测。

3. MMPP 突发标定
   两状态马尔可夫调制泊松过程（常态 λ0 / 突发态 λ1），由监控序列估计
   状态切换频率与突发态占比，输出突发态下的等效 ca²，供仿真和修正共用。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------- #
# 1. 变异系数修正
# ---------------------------------------------------------------------- #
def cv_squared(samples: Sequence[float]) -> float:
    """变异系数平方 = Var/Mean²（样本需为正，如到达间隔、服务时间）。"""
    x = np.asarray(samples, dtype=float)
    x = x[x > 0]
    if len(x) < 2 or np.mean(x) <= 0:
        return 1.0
    return float(np.var(x, ddof=1) / np.mean(x) ** 2)


def cv_dict(net, ca2_by_station: Optional[Dict[str, float]] = None,
            cs2_by_station: Optional[Dict[str, float]] = None
            ) -> Dict[str, Tuple[float, float]]:
    """组装 mva.solve 需要的 cv 参数；未指定的站点退回马尔可夫 (1,1)。"""
    out = {}
    for st in net.stations:
        out[st.name] = (
            float((ca2_by_station or {}).get(st.name, 1.0)),
            float((cs2_by_station or {}).get(st.name, 1.0)))
    return out


def allen_cunneen_correction(ca2: float, cs2: float = 1.0) -> float:
    """GI/GI/m 排队部分的放大系数。"""
    return 0.5 * (ca2 + cs2)


# ---------------------------------------------------------------------- #
# 2. 峰值区间修正
# ---------------------------------------------------------------------- #
@dataclass
class PeakResult:
    peak_lambda: float                 # 平滑后的峰值有效到达率（个/秒）
    mean_lambda: float
    peak_index: int                    # 峰值窗口序号
    burst_ratio: float                 # peak / mean
    window_rates: np.ndarray           # 各窗口原始到达率
    smoothed: np.ndarray               # 平滑后序列


def peak_interval_rate(counts_per_window: Sequence[float],
                       window_seconds: float,
                       smooth_windows: int = 3) -> PeakResult:
    """由各时间窗口的请求计数找出容量评估应采用的峰值到达率。

    采用滑动平均（SRS, simple random smoothing 思路的等权版）抑制单点
    抖动；``smooth_windows=1`` 时不平滑，直接取最不利窗口。
    """
    counts = np.asarray(counts_per_window, dtype=float)
    if window_seconds <= 0:
        raise ValueError("窗口时长必须为正")
    rates = counts / window_seconds
    if smooth_windows <= 1:
        smoothed = rates.copy()
    else:
        w = smooth_windows
        kernel = np.ones(w) / w
        # 边缘用截断平均，保持序列长度
        smoothed = np.array([
            np.mean(rates[max(0, i - w // 2): i + w // 2 + 1])
            for i in range(len(rates))])
    idx = int(np.argmax(smoothed))
    mean = float(np.mean(rates))
    peak = float(smoothed[idx])
    return PeakResult(peak, mean, idx,
                      peak / mean if mean > 0 else np.inf,
                      rates, smoothed)


def scale_open_arrivals(net, factor: float):
    """复制网络并把所有开放类到达率乘以 factor（闭合类不变）。"""
    return net.copy(arrival_scale=factor)


# ---------------------------------------------------------------------- #
# 3. MMPP 两状态突发标定
# ---------------------------------------------------------------------- #
@dataclass
class MMPP2:
    """两状态马尔可夫调制泊松过程。

    状态 0 常态、状态 1 突发态；r01/r10 为状态切换率（次/秒）。
    """
    lambda0: float
    lambda1: float
    r01: float
    r10: float

    @property
    def burst_fraction(self) -> float:
        """突发态长期占比。"""
        denom = self.r01 + self.r10
        return self.r01 / denom if denom > 0 else 0.0

    @property
    def mean_rate(self) -> float:
        p1 = self.burst_fraction
        return (1 - p1) * self.lambda0 + p1 * self.lambda1

    def equivalent_ca2(self, tau: float = 1.0) -> float:
        """MMPP-2 在时间尺度 τ（秒）上的等效到达间隔 ca²。

        突发态占比 p1、切换速率 r=r01+r10、调制方差 σ_λ²，
        计数过程在 τ 窗口上的方差-均值比为
            IDC(τ) = 1 + 2σ_λ²/(r·λ̄)·(1-(1-e^{-rτ})/(rτ))
        排队对突发的敏感时间尺度 τ 取站点平均忙周期量级（调用方传入），
        τ→0 时 IDC→1（退化为泊松），τ→∞ 时取长期调制方差。
        """
        p1 = self.burst_fraction
        r = self.r01 + self.r10
        lam_bar = self.mean_rate
        if r <= 0 or lam_bar <= 0:
            return 1.0
        var_lam = p1 * (1 - p1) * (self.lambda1 - self.lambda0) ** 2
        if r * tau < 1e-9:
            idc = 1.0
        else:
            idc = 1.0 + 2 * var_lam / (r * lam_bar) * (
                1.0 - (1.0 - np.exp(-r * tau)) / (r * tau))
        return max(1.0, float(idc))


def fit_mmpp2(counts_per_window: Sequence[float],
              window_seconds: float,
              quantile: float = 0.7) -> MMPP2:
    """由窗口计数序列用矩估计+分位数门限标定 MMPP-2。

    高于 q 分位数的窗口判为突发态，分别估计两态到达率、状态占比与
    平均驻留长度，进而换算切换率。
    """
    counts = np.asarray(counts_per_window, dtype=float)
    rates = counts / window_seconds
    if len(rates) < 4:
        return MMPP2(float(np.mean(rates)), float(np.mean(rates)),
                     1e-6, 1e-6)
    gate = float(np.quantile(rates, quantile))
    burst = rates >= gate
    p1 = float(np.mean(burst))
    lam0 = float(np.mean(rates[~burst])) if np.any(~burst) else gate
    lam1 = float(np.mean(rates[burst])) if np.any(burst) else gate
    # 平均驻留长度（窗口数）→ 切换率
    def _sojourn(mask: np.ndarray) -> float:
        lens: List[int] = []
        cur = 0
        for v in mask:
            if v:
                cur += 1
            elif cur:
                lens.append(cur)
                cur = 0
        if cur:
            lens.append(cur)
        return float(np.mean(lens)) if lens else 1.0
    d0, d1 = _sojourn(~burst) * window_seconds, _sojourn(burst) * window_seconds
    r10 = 1.0 / max(d1, window_seconds)
    r01 = p1 * r10 / max(1e-12, 1.0 - p1)
    return MMPP2(lam0, max(lam1, lam0 + 1e-9), r01, r10)


def burst_adjusted_cv(net, mmpp: MMPP2,
                      solution=None) -> Dict[str, Tuple[float, float]]:
    """把 MMPP 突发标定转换为各站点的 cv 字典。

    时间尺度 τ 取各 FCFS/PS 站点的平均忙周期估计 m/(μ(1-ρ))；
    无现成解时退化为单位时间尺度。
    """
    out: Dict[str, Tuple[float, float]] = {}
    for si, st in enumerate(net.stations):
        from .network import IS
        if st.discipline == IS:
            out[st.name] = (1.0, 1.0)
            continue
        if solution is not None and solution.util[si] < 0.999:
            D_mean = float(np.sum(solution.X * solution.net.demands[:, si]))
            a = solution.util[si] * st.servers
            tau = st.servers / max(D_mean, 1e-9) * 1.0 / max(
                1e-9, 1.0 - solution.util[si])
        else:
            tau = 1.0
        out[st.name] = (mmpp.equivalent_ca2(tau), 1.0)
    return out
