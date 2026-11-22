# -*- coding: utf-8 -*-
"""排队网络的解析求解。

实现三类算法
============
1. 开网络分析：多服务台 Erlang-C 公式，支持 GI/GI/m 的 Allen-Cunneen 修正。
2. 精确多类平均值分析（Exact MVA）：BCMP 乘积型网络的人口空间递推；
   多服务台站点在递推中使用到达定理形式的 Erlang-C 启发式（m=1 时严格精确）。
3. Schweitzer-Bard 近似 MVA：固定点迭代，复杂度 O(迭代数 × C × S)，
   用户数很大时替代精确递推（精确递推复杂度为 ∏(N_c+1)）。

关键恒等式（m 服务台，总负荷 a Erlang，ρ=a/m）
------------------------------------------------
Pwait = ErlangC(a, m)，排队队长 Lq = Pwait·ρ/(1-ρ)，
每类在站队长 L[c,s] = X[c]D[c,s]·(1 + Pwait/(m(1-ρ)))。
m=1 时 Pwait=ρ，因子退化为 1/(1-ρ)，与 M/M/1 及精确 MVA 完全一致。
IS（纯延迟）站点不排队：R[c,s]=D[c,s]。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .network import QueueNetwork, CLOSED, FCFS, PS, IS

EXACT_LATTICE_LIMIT = 5000      # 人口格点超过该规模则自动切换近似算法
DEFAULT_TOL = 1e-6


# ---------------------------------------------------------------------- #
# 数值工具
# ---------------------------------------------------------------------- #
def _logsumexp(logs: np.ndarray) -> float:
    m = np.max(logs)
    if not np.isfinite(m):
        return m
    return float(m + np.log(np.sum(np.exp(logs - m))))


def erlang_c(a: float, m: int) -> float:
    """Erlang-C 等待概率 P(wait>0)，到达负荷 a Erlang，m 个服务台。

    全部在对数域计算，避免 m 较大时阶乘溢出。``a`` 解释为到达者看到的
    平均在站负荷（MVA 到达定理口径），允许 a≥m：此时系统必发生等待，
    Pwait=1，由调用方按在站数口径计算排队，而非在此处发散。
    """
    if a <= 0.0:
        return 0.0
    if a >= m - 1e-12:
        return 1.0
    rho = a / m
    log_terms = np.array([k * math.log(a) - math.lgamma(k + 1.0)
                          for k in range(m)])
    log_last = (m * math.log(a) - math.lgamma(m + 1.0)
                - math.log(1.0 - rho))
    log_denom = _logsumexp(np.concatenate([log_terms, [log_last]]))
    return min(1.0, math.exp(log_last - log_denom))


def queue_factor(a: float, m: int, ca2: float = 1.0,
                 cs2: float = 1.0) -> Tuple[float, float, float]:
    """开网络口径：a 为 offered load（erlang），返回 (驻留因子, ρ, Pwait)。

    适用于开放网络和资源池 M/M/K。GI/GI/m 的 Allen-Cunneen 形式
    f = 1 + (ca²+cs²)/2 · Pwait/(m(1-ρ))。
    """
    if a <= 0.0:
        return 1.0, 0.0, 0.0
    k = 0.5 * (ca2 + cs2)
    if a >= m - 1e-12:
        f = 1.0 + k * (a - m + 1.0) / m
        return f, min(a / m, 1.0), 1.0
    rho = a / m
    pwait = erlang_c(a, m)
    f = 1.0 + k * pwait / (m * (1.0 - rho))
    return f, rho, pwait


def mva_seen_factor(seen: float, m: int, ca2: float = 1.0,
                    cs2: float = 1.0) -> float:
    """闭合 MVA 到达定理口径：seen 为到达者看到的平均 **在站顾客数**。

    在站顾客数可远大于服务台数 m，不能直接当作 erlang 负荷。把它分解为
    忙台部分与排队部分：等效 offered 负荷取 a=m·seen/(m+seen)（seen→∞
    时 a→m 但永不越界），等待部分用在站口径 seen/m。驻留因子

        f = 1 + (ca²+cs²)/2 · Pwait(a,m) · seen / m

    m=1 时 Pwait=a、代入后严格退化为精确 M/M/1 MVA 的 1+seen。
    """
    if seen <= 0.0:
        return 1.0
    k = 0.5 * (ca2 + cs2)
    if m == 1:
        return 1.0 + k * seen
    a_eff = m * seen / (m + seen)
    pwait = erlang_c(a_eff, m)
    return 1.0 + k * pwait * seen / m


# ---------------------------------------------------------------------- #
# 求解结果
# ---------------------------------------------------------------------- #
@dataclass
class Solution:
    net: QueueNetwork
    R: np.ndarray                  # 各类网络响应时间（不含思考时间），秒
    X: np.ndarray                  # 各类吞吐，个/秒
    util: np.ndarray               # 各站点利用率（IS 站点为负荷 erlang，非占比）
    L: np.ndarray                  # 各站点平均在站请求数
    Lq: np.ndarray                 # 各站点平均排队请求数
    L_class: np.ndarray            # [c,s] 各类在各站的平均请求数
    R_station: np.ndarray          # [c,s] 各类在各站的平均驻留时间
    pwait: np.ndarray              # 各站点等待概率
    stable: bool
    engine: str
    iterations: int
    notes: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"=== 求解结果（引擎={self.engine}, 稳定={self.stable}, "
                 f"迭代={self.iterations}）==="]
        lines.append("【请求类】 响应时间(s)  吞吐(/s)")
        for ci, c in enumerate(self.net.classes):
            lines.append(f"  {c.name:<12} {self.R[ci]:10.4f} "
                         f"{self.X[ci]:10.4f}")
        lines.append("【站点】 利用率  平均队长  排队队长  Pwait")
        for si, s in enumerate(self.net.stations):
            lines.append(f"  {s.name:<12} {self.util[si]:8.4f} "
                         f"{self.L[si]:9.4f} {self.Lq[si]:9.4f} "
                         f"{self.pwait[si]:7.4f}")
        return "\n".join(lines)


# ---------------------------------------------------------------------- #
# 主入口
# ---------------------------------------------------------------------- #
def solve(net: QueueNetwork,
          demands: Optional[np.ndarray] = None,
          cv: Optional[Dict[str, Tuple[float, float]]] = None,
          approx: str = "auto",
          tol: float = DEFAULT_TOL,
          max_iter: int = 500,
          think_override: Optional[np.ndarray] = None) -> Solution:
    """求解排队网络。

    参数
    ----
    demands: 可选的服务需求矩阵 D[c,s]，缺省用网络自身定义（反推/修正场景传入）。
    cv:      站点名 -> (到达间隔变异系数平方 ca2, 服务时间变异系数平方 cs2)，
             缺省 (1,1) 即马尔可夫假设；突发修正由 burst 模块计算后传入。
    approx:  "auto" / "exact" / "schweitzer"。
    think_override: 闭合类等效思考时间（分层模型把资源池等待并入此处，
             使其通过 X=N/(Z_eff+R) 自洽地压低闭合类吞吐，而不污染站点需求）。
    """
    D = net.demands if demands is None else np.asarray(demands, dtype=float)
    V = net.visits
    S = np.divide(D, V, out=np.zeros_like(D), where=V > 0)
    closed = net.closed_idx
    open_i = net.open_idx
    N = net.populations()
    think = (net.thinks() if think_override is None
             else np.asarray(think_override, dtype=float))
    lam = net.arrivals()
    cv = cv or {}
    ca2 = np.ones(net.S)
    cs2 = np.ones(net.S)
    for si, st in enumerate(net.stations):
        if st.name in cv:
            ca2[si], cs2[si] = cv[st.name]

    lattice = 1
    for c in closed:
        lattice *= int(N[c]) + 1
    use_exact = approx == "exact" or (approx == "auto"
                                      and lattice <= EXACT_LATTICE_LIMIT)
    if use_exact:
        try:
            return _exact_mva(net, D, S, closed, open_i, N, think, lam,
                              ca2, cs2, tol, max_iter)
        except _ExactUnsupported:
            pass
    return _schweitzer(net, D, S, closed, open_i, N, think, lam,
                       ca2, cs2, tol, max_iter)


class _ExactUnsupported(Exception):
    pass


# ---------------------------------------------------------------------- #
# 公共单轮固定点计算（layered 求解器也复用）
# ---------------------------------------------------------------------- #
def station_loads(net: QueueNetwork, D: np.ndarray, X: np.ndarray,
                  ca2: np.ndarray, cs2: np.ndarray,
                  seen_load: Optional[np.ndarray] = None
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """给定各类吞吐 X，计算每个站点的 (驻留因子, 利用率, Pwait, 负荷a)。

    seen_load 给定时用它代替 X·D 作为 Erlang-C 输入（到达定理视角，
    精确 MVA 递推时使用）。
    """
    f = np.ones(net.S)
    rho = np.zeros(net.S)
    pwait = np.zeros(net.S)
    a_full = np.zeros(net.S)
    for si, st in enumerate(net.stations):
        if st.discipline == IS:
            a_full[si] = np.sum(X * D[:, si])
            rho[si] = a_full[si]          # IS 报告负荷而非占比
            continue
        a = (np.sum(X * D[:, si]) if seen_load is None
             else seen_load[si])
        a_full[si] = a
        f[si], rho[si], pwait[si] = queue_factor(a, st.servers,
                                                 ca2[si], cs2[si])
    return f, rho, pwait, a_full


# ---------------------------------------------------------------------- #
# 精确多类 MVA（人口格点递推）
# ---------------------------------------------------------------------- #
def _exact_mva(net, D, S, closed, open_i, N, think, lam,
               ca2, cs2, tol, max_iter) -> Solution:
    # 开放类提供的固定 offered load（erlang）
    open_a = np.zeros(net.S)
    for c in open_i:
        open_a += lam[c] * D[c]

    # L_store[人口][闭合类k, 站s]；R_store 记录满人口点各类站级驻留
    L_store: Dict[Tuple[int, ...], np.ndarray] = {}
    X_store: Dict[Tuple[int, ...], np.ndarray] = {}
    R_store: Dict[Tuple[int, ...], np.ndarray] = {}

    def recurse(n_tuple: Tuple[int, ...]):
        if n_tuple in L_store:
            return
        if sum(n_tuple) == 0:
            L_store[n_tuple] = np.zeros((len(closed), net.S))
            X_store[n_tuple] = np.zeros(len(closed))
            R_store[n_tuple] = np.zeros((len(closed), net.S))
            return
        Lc = np.zeros((len(closed), net.S))
        Xc = np.zeros(len(closed))
        Rc = np.zeros((len(closed), net.S))
        for k, c in enumerate(closed):
            if n_tuple[k] == 0:
                continue
            prev = list(n_tuple)
            prev[k] -= 1
            prev_t = tuple(prev)
            recurse(prev_t)
            L_prev = L_store[prev_t]              # [k', s]，人口 n-e_c
            # 到达定理：类 c 的到达者看到的在站顾客数（erlang 负荷口径）
            # = 开放类负荷 + 所有闭合类在 n-e_c 下的平均在站数之和。
            seen = open_a.copy()
            for k2 in range(len(closed)):
                seen += L_prev[k2]
            x = 0.0
            for si, st in enumerate(net.stations):
                if D[c, si] <= 0:
                    continue
                if st.discipline == IS:
                    Rc[k, si] = D[c, si]
                else:
                    f = mva_seen_factor(seen[si], st.servers,
                                        ca2[si], cs2[si])
                    Rc[k, si] = D[c, si] * f
            R_total = float(np.sum(Rc[k]))
            x = n_tuple[k] / (think[c] + R_total) \
                if math.isfinite(R_total) else 0.0
            Xc[k] = x
            for si in range(net.S):
                if D[c, si] <= 0:
                    continue
                Lc[k, si] = x * Rc[k, si]
        L_store[n_tuple] = Lc
        X_store[n_tuple] = Xc
        R_store[n_tuple] = Rc

    full = tuple(int(N[c]) for c in closed)
    recurse(full)

    # 组装全类结果（直接采用递归在满人口点的驻留时间，不再二次重算）
    X = np.zeros(net.C)
    R_station = np.zeros((net.C, net.S))
    L_class = np.zeros((net.C, net.S))
    for c in open_i:
        X[c] = lam[c]
    for k, c in enumerate(closed):
        X[c] = X_store[full][k]
        R_station[c] = R_store[full][k]
        L_class[c] = L_store[full][k]
    # 开放类站级指标：用满人口点的闭合在站负荷 + 开放 offered load
    open_seen = np.zeros(net.S)
    for k2 in range(len(closed)):
        open_seen += L_store[full][k2]
    for c in open_i:
        for si, st in enumerate(net.stations):
            if D[c, si] <= 0:
                continue
            if st.discipline == IS:
                R_station[c, si] = D[c, si]
            else:
                seen_load = open_seen[si] + np.sum(
                    lam[o] * D[o, si] for o in open_i)
                f = mva_seen_factor(seen_load, st.servers,
                                    ca2[si], cs2[si])
                R_station[c, si] = D[c, si] * f
        L_class[c] = X[c] * R_station[c]
    return _assemble_from_parts(
        net, D, X, R_station, L_class, ca2, cs2,
        engine="exact-mva", iterations=1,
        notes=[f"人口格点数 {int(np.prod(np.array(full) + 1))}"])


# ---------------------------------------------------------------------- #
# Schweitzer-Bard 近似 MVA（固定点迭代，任意服务台数）
# ---------------------------------------------------------------------- #
def _schweitzer(net, D, S, closed, open_i, N, think, lam,
                ca2, cs2, tol, max_iter) -> Solution:
    open_a = np.zeros(net.S)
    for c in open_i:
        open_a += lam[c] * D[c]

    # 初值：假设无排队
    X = np.zeros(net.C)
    for c in open_i:
        X[c] = lam[c]
    Lc = np.zeros((net.C, net.S))
    for c in closed:
        X[c] = N[c] / (think[c] + np.sum(D[c]))
        for si in range(net.S):
            Lc[c, si] = X[c] * D[c, si]

    notes: List[str] = []
    it = 0
    for it in range(1, max_iter + 1):
        X_new = X.copy()
        for c in closed:
            # 到达定理的 Schweitzer 比例估计
            scale = (N[c] - 1.0) / N[c] if N[c] > 0 else 0.0
            seen = open_a.copy()
            for c2 in closed:
                seen += (Lc[c2] * scale) if c2 == c else Lc[c2]
            R_c = 0.0
            for si, st in enumerate(net.stations):
                if D[c, si] <= 0:
                    continue
                if st.discipline == IS:
                    R_c += D[c, si]
                else:
                    f = mva_seen_factor(seen[si], st.servers,
                                        ca2[si], cs2[si])
                    R_c += D[c, si] * f
            if not math.isfinite(R_c):
                notes.append(f"类 {net.classes[c].name} 出现过载，迭代提前终止")
                X_new[c] = 0.0
                continue
            X_new[c] = N[c] / (think[c] + R_c)
            for si in range(net.S):
                if D[c, si] <= 0:
                    continue
                st = net.stations[si]
                if st.discipline == IS:
                    Lc[c, si] = X_new[c] * D[c, si]
                else:
                    f = mva_seen_factor(seen[si], st.servers,
                                        ca2[si], cs2[si])
                    Lc[c, si] = X_new[c] * D[c, si] * f
        # 松弛防止饱和点附近振荡
        X_new = 0.5 * X + 0.5 * X_new
        if np.all(np.abs(X_new[closed] - X[closed])
                  <= tol * np.maximum(X_new[closed], 1e-12)):
            X = X_new
            break
        X = X_new
    else:
        notes.append("Schweitzer 迭代达到上限未完全收敛")

    sol = _assemble(net, D, X, ca2, cs2, engine="schweitzer-mva",
                    iterations=it, notes=notes)
    if not sol.stable:
        sol.notes.append("存在站点利用率≥100%，结果为过载外推，不可用于容量承诺")
    return sol


# ---------------------------------------------------------------------- #
# 结果组装
# ---------------------------------------------------------------------- #
def _assemble_from_parts(net: QueueNetwork, D: np.ndarray, X: np.ndarray,
                         R_station: np.ndarray, L_class: np.ndarray,
                         ca2: np.ndarray, cs2: np.ndarray, engine: str,
                         iterations: int, notes: List[str]) -> Solution:
    """由求解器已确定的站级驻留时间与在站数直接组装（口径自洽）。

    利用率/负荷由 Little 定律 L = Σ_c L_class[c,s] 给出：
    排队站利用率 = 忙台数/m = (L - Lq)/m，其中 Lq 为超出 m 的在站部分；
    IS 站报告平均在场数（erlang 负荷）。
    """
    R = np.sum(R_station, axis=1)
    L = np.sum(L_class, axis=0)
    util = np.zeros(net.S)
    Lq = np.zeros(net.S)
    pwait = np.zeros(net.S)
    for si, st in enumerate(net.stations):
        if st.discipline == IS:
            util[si] = L[si]
            continue
        # 服务定律：X·D 即忙台占比（D 为每轮总服务时间，已含 1/m 口径）
        a = float(np.sum(X * D[:, si]))
        util[si] = min(a, 1.0)
        Lq[si] = max(0.0, L[si] - a * st.servers)
        pwait[si] = erlang_c(a * st.servers, st.servers)
    stable = bool(np.all(util[[si for si, s in enumerate(net.stations)
                               if s.discipline != IS]] < 1.0))
    return Solution(net=net, R=R, X=X, util=util, L=L, Lq=Lq,
                    L_class=L_class, R_station=R_station,
                    pwait=pwait, stable=stable, engine=engine,
                    iterations=iterations, notes=notes)


def _assemble(net: QueueNetwork, D: np.ndarray, X: np.ndarray,
              ca2: np.ndarray, cs2: np.ndarray, engine: str,
              iterations: int, notes: List[str]) -> Solution:
    f, rho, pwait, a = station_loads(net, D, X, ca2, cs2)
    R_station = D * f[np.newaxis, :]
    R = np.sum(R_station, axis=1)
    # 闭合类吞吐必须与固定点迭代中的 X 保持自洽：X=N/(Z+R)。
    # 这里由刚算出的 R 做一次最终自洽校正，并同步重算各类队长；
    # 开放类吞吐保持外生到达率。
    X = X.copy()
    for c in net.closed_idx:
        X[c] = (net.populations()[c] / (net.thinks()[c] + R[c])
                if math.isfinite(R[c]) else 0.0)
    L_class = (X * D.T).T * f[np.newaxis, :]
    L = np.sum(L_class, axis=0)
    Lq = np.maximum(L - a, 0.0)
    stable = bool(np.all(rho[[si for si, s in enumerate(net.stations)
                              if s.discipline != IS]] < 1.0))
    return Solution(net=net, R=R, X=X, util=rho, L=L, Lq=Lq,
                    L_class=L_class, R_station=R_station,
                    pwait=pwait, stable=stable, engine=engine,
                    iterations=iterations, notes=notes)
