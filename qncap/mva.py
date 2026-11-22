"""解析求解器: 平均值分析 (Mean Value Analysis)。

两条求解路线:

* :func:`solve_exact` —— 多闭合链精确 MVA。按人口向量格点递归,
  每个多服务台站点保存边际占用分布, 精确给出类响应时间、吞吐、利用率。
  复杂度随人口格点数增长, 用户数很大时变慢;
* :func:`solve_approx` —— Bard-Schweitzer 近似迭代 MVA。
  用"到达定理 + 比例缩放"在固定人口上做不动点迭代, 复杂度与
  用户数无关, 适合大人口与混合开闭网络(开放类作为背景负载并入)。

口径: 嵌套调用先由 ``layered`` 模块展开或等效化, 本模块在
需求空间 D[k,i](每次事务在站点 i 的总服务需求, 秒)上工作。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import product
from typing import Dict, List, Optional, Tuple

import numpy as np

from .network import Network, SCHED_IS


@dataclass
class ModelResult:
    """解析模型输出。"""
    network_name: str
    method: str
    class_response: Dict[str, float] = field(default_factory=dict)
    class_throughput: Dict[str, float] = field(default_factory=dict)
    class_population: Dict[str, float] = field(default_factory=dict)
    station_utilization: Dict[str, float] = field(default_factory=dict)
    station_throughput: Dict[str, float] = field(default_factory=dict)
    station_queue_length: Dict[str, float] = field(default_factory=dict)
    station_residence: Dict[str, float] = field(default_factory=dict)
    per_class_residence: Dict[str, Dict[str, float]] = field(default_factory=dict)
    per_class_queue: Dict[str, Dict[str, float]] = field(default_factory=dict)
    stable: bool = True
    iterations: int = 0
    max_rel_change: float = 0.0
    warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------- #
# 公共工具
# ---------------------------------------------------------------------- #
def _demand_arrays(net: Network
                   ) -> Tuple[List[str], List[str], np.ndarray, np.ndarray]:
    snames = list(net.stations.keys())
    cnames = list(net.classes.keys())
    dem = net.per_class_demands()
    vis = net.per_class_visits()
    D = np.array([[dem[c][s] for s in snames] for c in cnames], dtype=float)
    V = np.array([[vis[c][s] for s in snames] for c in cnames], dtype=float)
    return snames, cnames, D, V


def _servers_array(net: Network, snames: List[str]) -> np.ndarray:
    return np.array([
        net.stations[s].servers if net.stations[s].servers is not None
        else np.inf for s in snames], dtype=float)


def erlang_c(m: int, a: float) -> float:
    """Erlang-C: m 台、提供负载 a(Erlang)时到达必须等待的概率。

    全程对数空间计算, 避免大 m 下 a**m / factorial(m) 溢出。
    """
    if a >= m - 1e-12:
        return 1.0
    if a <= 0:
        return 0.0
    # log( sum_{k=0}^{m-1} a^k/k! ) 用 log-sum-exp
    log_terms = [0.0]  # k=0 项 = 1, log=0
    log_fact = 0.0
    for k in range(1, m):
        log_fact += math.log(k)
        log_terms.append(k * math.log(a) - log_fact)
    lse = max(log_terms)
    log_sum = lse + math.log(sum(math.exp(t - lse) for t in log_terms))
    # log(尾部项 a^m / (m! (1-a/m)))
    log_fact_m = log_fact + math.log(m)
    log_tail = (m * math.log(a) - log_fact_m
                - math.log1p(-a / m))
    # Pwait = tail / (sum + tail) = 1 / (1 + sum/tail)
    log_sum_over_tail = log_sum - log_tail
    if log_sum_over_tail > 700:        # exp 上溢保护: tail 可忽略
        return 0.0
    if log_sum_over_tail < -700:       # exp 下溢保护: sum 可忽略
        return 1.0
    return 1.0 / (1.0 + math.exp(log_sum_over_tail))


def mm_m_wait(m: int, rate: float, s_eff: float) -> float:
    """M/M/m 平均排队等待时间(不含服务)。"""
    if rate <= 0 or s_eff <= 0:
        return 0.0
    a = rate * s_eff
    if a >= m:
        return math.inf
    return erlang_c(m, a) * s_eff / (m * (1.0 - a / m))


def _blended_service(D: np.ndarray, V: np.ndarray, idxs: List[int],
                     j: int, rates: np.ndarray) -> float:
    """按站点到达率加权的平均单次服务时间。"""
    num = den = 0.0
    for i in idxs:
        lam = rates[i] * V[i, j]
        if lam > 0 and V[i, j] > 0:
            num += lam * (D[i, j] / V[i, j])
            den += lam
    return num / den if den > 0 else 0.0


# ---------------------------------------------------------------------- #
# 近似 MVA: Bard-Schweitzer (开放类背景负载 + 多服务台)
# ---------------------------------------------------------------------- #
def solve_approx(net: Network, max_iter: int = 500,
                 tol: float = 1e-7,
                 open_rates: Optional[Dict[str, float]] = None,
                 ) -> ModelResult:
    snames, cnames, D, V = _demand_arrays(net)
    m = _servers_array(net, snames)
    is_is = np.array([net.stations[s].sched == SCHED_IS for s in snames])
    closed = [i for i, c in enumerate(cnames)
              if net.classes[c].kind == "closed"]
    open_ = [i for i, c in enumerate(cnames)
             if net.classes[c].kind == "open"]
    N = np.array([net.classes[c].population or 0 for c in cnames],
                 dtype=float)
    Z = np.array([net.classes[c].think_time for c in cnames])
    open_rate = np.zeros(len(cnames))
    for i in open_:
        open_rate[i] = (open_rates or {}).get(
            cnames[i], net.classes[cnames[i]].arrival_rate)

    K, S = D.shape
    lam_open = open_rate @ V
    Qc = np.zeros((K, S))
    for i in closed:
        weight = D[i] + 1e-9
        Qc[i] = N[i] * weight / weight.sum()

    res = ModelResult(network_name=net.name, method="Bard-Schweitzer MVA")
    R_final = np.zeros((K, S))
    X_final = np.zeros(K)
    total_closed_pop = int(sum(N[i] for i in closed)) if closed else 0
    # 上一轮闭合类吞吐(首轮用 N/(Z+需求下界) 作初值)
    Xc_prev = np.array([
        (N[i] / (Z[i] + D[i].sum()) if i in closed else 0.0)
        for i in range(K)])

    for it in range(1, max_iter + 1):
        # 当前迭代各站点的闭合类在站人口(到达定理的输入)
        pop_at = Qc.sum(axis=0)
        # 1) 闭合类驻留
        R_new = np.zeros((K, S))
        station_lq = np.zeros(S)
        for j in range(S):
            if is_is[j]:
                for i in closed:
                    R_new[i, j] = D[i, j]
                continue
            mj = int(m[j])
            # 用上一轮真实吞吐把多类需求压成加权平均单次服务时间
            s_blended = _blended_service(D, V, closed, j, Xc_prev)
            a_open = 0.0
            if lam_open[j] > 0:
                s_o = _blended_service(D, V, open_, j, open_rate)
                a_open = lam_open[j] * s_o
            n_here = max(int(round(pop_at[j])), 0)
            if mj <= 1:
                # 单台 M/M/1: 到达者看到的在站人数(BS 缩放), 站点等待
                # 时间对所有类相同 = 看到人数 * 站点平均单次服务时间
                s_station = _blended_service(D, V, closed + open_, j,
                                             np.where(
                                                 np.isin(range(K), closed),
                                                 Xc_prev, open_rate))
                w_open_1 = 0.0
                if lam_open[j] > 0:
                    s_o = _blended_service(D, V, open_, j, open_rate)
                    w_open_1 = mm_m_wait(1, lam_open[j], s_o)
                for i in closed:
                    q_seen = sum(
                        Qc[h, j] * ((N[h] - 1.0) / max(N[h], 1.0)
                                    if h == i else 1.0)
                        for h in closed)
                    w_j = q_seen * s_station + w_open_1
                    R_new[i, j] = D[i, j] + V[i, j] * w_j
                station_lq[j] = pop_at[j]
            else:
                # 多台: 由当前迭代的站点到达率(闭合+开放)用 M/M/m。
                # 到达率 X_k*V_kj 已由 MVA 不动点给出, 自带 think 时间效应。
                lam_c = sum((Xc_prev[h] if h in closed else 0.0)
                            * V[h, j] for h in range(K))
                lam_tot_here = lam_c + lam_open[j]
                if lam_tot_here > 0 and s_blended > 0:
                    w_j = mm_m_wait(mj, lam_tot_here, s_blended)
                    if not math.isfinite(w_j):
                        # 饱和: 闭合系统人口有限, 用有限在站人口给有界等待,
                        # 而非开放 M/M/m 的无限发散(会产生 1e9 哨兵污染)
                        n_present = max(pop_at[j], float(mj))
                        w_j = max(n_present - mj, 0.0) * s_blended
                    exp_q = lam_tot_here * w_j
                else:
                    w_j = 0.0
                    exp_q = 0.0
                station_lq[j] = exp_q
                for i in closed:
                    # 站点每次访问平均等待 w_j 对各类相同, 按访问次数计价
                    R_new[i, j] = D[i, j] + V[i, j] * w_j

        # 2) 闭合类吞吐与队列
        Xc = np.zeros(K)
        Q_new = np.zeros((K, S))
        for i in closed:
            total_R = R_new[i].sum()
            Xc[i] = N[i] / (Z[i] + total_R) if total_R > 0 else 0.0
            Q_new[i] = Xc[i] * R_new[i]

        # 3) 开放类驻留: 合并负载下的 Erlang-C
        R_open = np.zeros((K, S))
        rates_all = np.array([
            Xc[i] if i in closed else open_rate[i] for i in range(K)])
        for j in range(S):
            if is_is[j]:
                for i in open_:
                    R_open[i, j] = D[i, j]
                continue
            mj = int(m[j])
            lam_tot = float(rates_all @ V[:, j])
            s_eff = _blended_service(D, V, closed + open_, j, rates_all)
            w = mm_m_wait(mj, lam_tot, s_eff)
            if not math.isfinite(w):
                if res.stable:
                    res.stable = False
                    res.warnings.append(
                        f"站点 {snames[j]} 过载(到达负载>=服务台数), "
                        "开放类排队发散, 下列响应时间为受限估计下界")
                # 用一个明确的大有限值而非 inf/1e9, 保持结果可读
                w = 1e3
            for i in open_:
                if V[i, j] > 0:
                    R_open[i, j] = V[i, j] * (D[i, j] / V[i, j] + w)

        R_final = R_new + R_open
        X_final = Xc
        res.iterations = it
        diff = (np.abs(Q_new - Qc).sum() / (Q_new.sum() + 1e-12)
                if closed else 0.0)
        res.max_rel_change = diff
        Qc = Q_new
        Xc_prev = Xc.copy()
        if diff < tol:
            break
    else:
        res.warnings.append("近似 MVA 未在最大迭代次数内收敛")

    return _build_result(net, snames, cnames, D, V, m, is_is,
                         R_final, X_final, open_rate, res,
                         method="Bard-Schweitzer MVA")


# ---------------------------------------------------------------------- #
# 精确多链 MVA(仅闭合类; 多服务台边际占用分布递归)
# ---------------------------------------------------------------------- #
def solve_exact(net: Network) -> ModelResult:
    """多闭合链精确 MVA(Reiser 递归的多服务台 load-dependent 版本)。

    仅支持纯闭合网络; 含开放类请用 :func:`solve_approx`。
    每个站点在每个人口向量下保存边际占用分布 P_i(n|N):

    * R[k,j](N): 到达定理, 用 N-e_k 下的期望队列长;
    * X[k](N) = N[k] / (Z[k] + sum_j R[k,j]);
    * P_i(n|N) = (1/mu_i(n)) * sum_k D[k,i]*X[k](N-e_k) * P_i(n-1|N-e_k),
      其中 mu_i(n) = min(n, m_i)(单位速率服务台)。
    """
    snames, cnames, D, V = _demand_arrays(net)
    if any(net.classes[c].kind == "open" for c in cnames):
        raise ValueError("精确 MVA 仅支持闭合类网络; 含开放类请用 solve_approx")
    m = _servers_array(net, snames)
    is_is = np.array([net.stations[s].sched == SCHED_IS for s in snames])
    K, S = D.shape
    Nv = np.array([net.classes[c].population for c in cnames], dtype=int)
    Z = np.array([net.classes[c].think_time for c in cnames])

    X_store: Dict[Tuple[int, ...], np.ndarray] = {}
    R_store: Dict[Tuple[int, ...], np.ndarray] = {}
    P_store: Dict[Tuple[int, ...], List[np.ndarray]] = {}
    zero = tuple(0 for _ in range(K))
    X_store[zero] = np.zeros(K)
    R_store[zero] = np.zeros((K, S))
    P_store[zero] = [np.array([1.0]) for _ in range(S)]

    vectors = _all_population_vectors(Nv)
    for vec in vectors[1:]:
        nvec = np.array(vec, dtype=float)
        R = np.zeros((K, S))
        for k in range(K):
            if vec[k] == 0:
                continue
            prev = tuple(vec[i] - (1 if i == k else 0) for i in range(K))
            for j in range(S):
                if is_is[j]:
                    R[k, j] = D[k, j]
                    continue
                mj = int(m[j])
                s_kj = D[k, j] / V[k, j] if V[k, j] > 0 else 0.0
                dist = P_store[prev][j]
                exp_q = sum(max(n - mj, 0) * dist[n]
                            for n in range(len(dist)))
                R[k, j] = D[k, j] + exp_q * s_kj
        X = np.array([
            nvec[k] / (Z[k] + R[k].sum()) if vec[k] > 0 else 0.0
            for k in range(K)])

        cap = int(sum(vec))
        dists: List[np.ndarray] = []
        for j in range(S):
            if is_is[j]:
                mean_n = float(sum(X[k] * D[k, j] for k in range(K)))
                size = max(int(mean_n * 4) + 2, 2)
                p = np.array([
                    math.exp(-mean_n) * mean_n ** n / math.factorial(n)
                    for n in range(size)])
                dists.append(p / p.sum())
                continue
            mj = int(m[j])
            terms = [(k, X[k] * D[k, j],
                      P_store[tuple(vec[i] - (1 if i == k else 0)
                                   for i in range(K))][j])
                     for k in range(K)
                     if vec[k] > 0 and D[k, j] > 0]
            pnew = np.zeros(cap + 1)
            for n in range(1, cap + 1):
                rate_n = min(n, mj)
                acc = 0.0
                for _, xkd, dist in terms:
                    if n - 1 < len(dist):
                        acc += xkd * dist[n - 1]
                pnew[n] = acc / rate_n
            pnew[0] = max(1.0 - pnew[1:].sum(), 0.0)
            total = pnew.sum()
            if not math.isfinite(total) or total <= 0:
                pnew = np.array([1.0])
            else:
                pnew = pnew / total
                tail = np.cumsum(pnew[::-1])
                keep = int(np.searchsorted(tail, 1e-12))
                if 1 < keep < len(pnew) - 1:
                    pnew = pnew[:keep + 1]
                    pnew = pnew / pnew.sum()
            dists.append(pnew)
        X_store[vec] = X
        R_store[vec] = R
        P_store[vec] = dists

    final = tuple(int(x) for x in Nv)
    res = ModelResult(network_name=net.name, method="exact MVA",
                      iterations=len(vectors))
    return _build_result(net, snames, cnames, D, V, m, is_is,
                         R_store[final], X_store[final],
                         np.zeros(K), res, method="exact MVA")


def _all_population_vectors(nv: np.ndarray) -> List[Tuple[int, ...]]:
    vecs = list(product(*[range(n + 1) for n in nv]))
    vecs.sort(key=lambda v: (sum(v), v))
    return vecs


# ---------------------------------------------------------------------- #
# 结果组装
# ---------------------------------------------------------------------- #
def _build_result(net: Network, snames: List[str], cnames: List[str],
                  D: np.ndarray, V: np.ndarray, m: np.ndarray,
                  is_is: np.ndarray, R: np.ndarray, Xc: np.ndarray,
                  open_rate: np.ndarray, res: ModelResult,
                  method: str) -> ModelResult:
    K, S = D.shape
    X = Xc.copy()
    for i, c in enumerate(cnames):
        if net.classes[c].kind == "open":
            X[i] = open_rate[i]

    for i, c in enumerate(cnames):
        res.class_response[c] = float(R[i].sum())
        res.class_throughput[c] = float(X[i])
        res.class_population[c] = float(
            net.classes[c].population if net.classes[c].kind == "closed"
            else X[i] * R[i].sum())
        res.per_class_residence[c] = {
            snames[j]: float(R[i, j]) for j in range(S)}
        res.per_class_queue[c] = {
            snames[j]: float(X[i] * R[i, j]) for j in range(S)}

    for j, s in enumerate(snames):
        lam = float(sum(X[i] * V[i, j] for i in range(K)))
        work = float(sum(X[i] * D[i, j] for i in range(K)))
        st = net.stations[s]
        if is_is[j]:
            util = work                       # IS: 平均占用台数
        else:
            util = work / st.servers
            if util > 0.999 and res.stable:
                res.stable = False
                res.warnings.append(
                    f"站点 {s} 利用率 {util:.3f} 达到/超过容量上限")
            util = min(util, 1.0)
        res.station_utilization[s] = util
        res.station_throughput[s] = lam
        res.station_queue_length[s] = float(
            sum(X[i] * R[i, j] for i in range(K)))
        res.station_residence[s] = (
            res.station_queue_length[s] / lam if lam > 0 else 0.0)
    res.method = method
    return res


def solve(net: Network, method: str = "auto", **kwargs) -> ModelResult:
    """统一入口: 'exact' / 'approx' / 'auto'。"""
    has_open = any(c.kind == "open" for c in net.classes.values())
    if method == "exact":
        return solve_exact(net, **kwargs)
    if method == "approx":
        return solve_approx(net, **kwargs)
    if has_open:
        return solve_approx(net, **kwargs)
    lattice = 1
    for c in net.classes.values():
        if c.kind == "closed":
            lattice *= (c.population + 1)
    return (solve_exact(net, **kwargs) if lattice <= 20000
            else solve_approx(net, **kwargs))
