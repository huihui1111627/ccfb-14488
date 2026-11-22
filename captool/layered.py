# -*- coding: utf-8 -*-
"""分层排队网络（Layered Queueing）求解器。

问题背景
========
真实系统里连接池、线程池是容量有限的资源，且请求会“拿着一层的资源
去等下一层”：例如应用线程持有期间去申请数据库连接，连接池满时线程
被阻塞但仍不释放，导致上层资源被无效占用。扁平的 BCMP 网络无法表达
这种嵌套占用。

建模方法（流等效 / Flow-Equivalent 分层迭代）
=============================================
1. 把每个资源池展开为一个虚拟的 FCFS 多服务台站点：
   - 服务台数 = 池容量 K（令牌数）；
   - 每次访问的“服务时间” = 令牌持有时间 H，即从 acquire 到 release
     之间经过的全部服务步时间之和；
   - 嵌套池的排队等待发生在持有外层令牌期间，因此内层等待要计入
     外层令牌的持有时间（内层等待 → 外层 H 膨胀）。
2. 在展开网络上调用 :mod:`captool.mva`，由 Erlang-C/MVA 得到各池的
   平均等待 W_p、利用率和吞吐。
3. 用新的 W_p 更新外层持有时间，重复直到 W 与 X 不动点收敛。
4. 池的到达率 λ_p = Σ_c X[c]·v_p[c]（v 为每轮申请次数），
   负荷 a_p = Σ_c X[c]·v_p[c]·H_eff[c]；a_p ≥ K_p 即池饱和。

该方法是 LQN 文献中 flow-equivalent server 近似的工程实现：指数服务
假设下内层单池为 M/M/K，多层嵌套经不动点迭代逼近。仿真验收见
:mod:`captool.validate`。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .network import QueueNetwork, FCFS
from . import mva


class _SolutionWithRT:
    """包装站点层 MVA 解，仅用“站点驻留+池等待”的端到端 R 覆盖原 R。

    其余指标（吞吐、利用率、队长、网络引用等）全部委托给内层站点解，
    保证站点口径不被池等待污染。
    """

    def __init__(self, inner: "mva.Solution", rt_total: np.ndarray):
        self._inner = inner
        self.R = np.asarray(rt_total, dtype=float)

    def __getattr__(self, item):
        return getattr(self._inner, item)


@dataclass
class LayeredSolution:
    base: mva.Solution                 # 展开网络（含虚拟池站点）的解
    pool_names: List[str]
    pool_servers: np.ndarray           # 各池容量
    pool_load: np.ndarray              # 各池负荷（erlang）
    pool_util: np.ndarray              # 各池利用率 = 负荷/容量
    pool_wait: np.ndarray              # 各池每次申请的平均排队等待（秒）
    pool_queue: np.ndarray             # 各池平均等待令牌数
    original_S: int                    # 原网络站点数（其后为虚拟池站点）
    iterations: int
    converged: bool
    notes: List[str]

    # ---------------------------------------------------------------- #
    # 对原网络口径的结果访问（base 只含原站点，池指标单独存放）
    # ---------------------------------------------------------------- #
    @property
    def R(self) -> np.ndarray:
        """各类端到端响应时间（秒，已含所有池等待）。"""
        return self.base.R

    @property
    def X(self) -> np.ndarray:
        return self.base.X

    @property
    def util(self) -> np.ndarray:
        """原网络各站点利用率。"""
        return self.base.util

    @property
    def L(self) -> np.ndarray:
        return self.base.L

    @property
    def Lq(self) -> np.ndarray:
        return self.base.Lq

    @property
    def pwait(self) -> np.ndarray:
        return self.base.pwait

    @property
    def stable(self) -> bool:
        return self.base.stable and bool(np.all(self.pool_util < 1.0))

    def station_view(self) -> List[dict]:
        """原站点 + 资源池统一口径的容量视图（供页面与报告使用）。"""
        net = self.base.net
        rows = []
        for si, st in enumerate(net.stations):
            rows.append({
                "name": st.name, "kind": "station",
                "servers": st.servers, "discipline": st.discipline,
                "util": float(self.base.util[si]),
                "queue": float(self.base.Lq[si]),
                "pwait": float(self.base.pwait[si]),
            })
        for k, pname in enumerate(self.pool_names):
            rows.append({
                "name": pname, "kind": "pool",
                "servers": int(self.pool_servers[k]),
                "discipline": FCFS,
                "util": float(self.pool_util[k]),
                "queue": float(self.pool_queue[k]),
                "pwait": float(mva.erlang_c(self.pool_load[k],
                                            int(self.pool_servers[k]))),
                "wait": float(self.pool_wait[k]),
            })
        return rows

    def summary(self) -> str:
        lines = [f"=== 分层模型结果（迭代 {self.iterations} 次，"
                 f"收敛={self.converged}，稳定={self.stable}）==="]
        lines.append(self.base.summary())
        lines.append("【资源池】 容量  利用率  平均等待(s)  等待队列")
        for k, pname in enumerate(self.pool_names):
            lines.append(f"  {pname:<12} {int(self.pool_servers[k]):4d} "
                         f"{self.pool_util[k]:8.4f} {self.pool_wait[k]:12.4f} "
                         f"{self.pool_queue[k]:10.4f}")
        if self.notes:
            lines.append("备注: " + "; ".join(self.notes))
        return "\n".join(lines)


class LayeredSolver:
    """分层排队网络求解器。"""

    def __init__(self, net: QueueNetwork, tol: float = 1e-5,
                 max_iter: int = 100, relax: float = 0.5):
        self.net = net
        self.tol = tol
        self.max_iter = max_iter
        self.relax = relax
        self._build()

    # ------------------------------------------------------------------ #
    # 池结构索引：申请次数、容量、裸持有时间与嵌套引用
    # ------------------------------------------------------------------ #
    def _build(self):
        net = self.net
        self.pool_names = [p.name for p in net.pools]
        # 每类在各池的每轮申请次数
        self.acq_visits = net.pool_acquire_visits()        # dict[p] -> [C]
        self.capacities = np.array([p.capacity for p in net.pools],
                                   dtype=float)
        # 池持有区间内的“裸服务时间”（不含内层池等待），按每次申请计
        self._bare_hold = self._bare_holding_times()
        # 持有区间内出现的内层池申请：(外池 p, 内池 q) -> 每类每轮次数
        self._inner_refs = self._inner_references()

    def _bare_holding_times(self) -> Dict[str, np.ndarray]:
        """每个池、每类、每次申请的平均裸持有时间（区间内服务时间之和）。"""
        net = self.net
        out = {p: np.zeros(net.C) for p in self.pool_names}
        for ci, c in enumerate(net.classes):
            held: List[str] = []
            for step in c.route:
                if step.acquire:
                    held.append(step.acquire)
                for pname in held:
                    out[pname][ci] += step.service
                if step.release:
                    held.remove(step.release)
        # 除以每类对该池的申请次数，得到“每次申请”口径
        for pname in out:
            v = self.acq_visits[pname]
            out[pname] = np.divide(out[pname], v, out=out[pname],
                                   where=v > 0)
        return out

    def _inner_references(self) -> Dict[str, Dict[str, np.ndarray]]:
        """外池持有区间内对内池的申请次数（每类每轮）。"""
        net = self.net
        refs: Dict[str, Dict[str, np.ndarray]] = {
            p: {q: np.zeros(net.C) for q in self.pool_names if q != p}
            for p in self.pool_names}
        for ci, c in enumerate(net.classes):
            held: List[str] = []
            for step in c.route:
                if step.acquire:
                    for outer in held:
                        refs[outer][step.acquire][ci] += 1.0
                    held.append(step.acquire)
                if step.release:
                    held.remove(step.release)
        return refs

    # ------------------------------------------------------------------ #
    # 不动点迭代（Erlang-C 池等待 + 等效思考时间）
    # ------------------------------------------------------------------ #
    def solve(self, approx: str = "auto",
              cv: Optional[Dict[str, tuple]] = None) -> LayeredSolution:
        """分层排队网络不动点求解。

        语义
        ====
        资源池令牌在某路由步 acquire、在后续步 release。请求若拿不到令牌，
        会在池队列中阻塞，期间 **不占用** 内层任何站点的服务台，只继续
        持有已获得的外层令牌。因此：

        - 站点的有效服务需求始终是裸服务时间，池等待 **不** 加进站点需求；
        - 池等价为 M/M/K：容量 K、申请率 λ_p=Σ_c X[c]v_p[c]、服务时间为
          令牌有效持有时间 H_eff（裸持有 + 嵌套内层池等待），等待由
          Erlang-C 给出；
        - 对闭合类，池等待 Wtot[c]=Σ_p v_p[c]W_p 等价为额外“思考时间”，
          通过 X=N/(Z+Wtot+R_station) 自洽压低吞吐；
        - 对开放类，到达率外生，池等待直接叠加到端到端响应时间。
        """
        net = self.net
        D_base = net.demands
        base_think = net.thinks()
        notes: List[str] = []

        W = {p: 0.0 for p in self.pool_names}
        prev_W = {p: -1.0 for p in self.pool_names}
        prev_X = None
        converged = False
        sol = None
        it = 0
        for it in range(1, self.max_iter + 1):
            # 1) 各类每轮的池等待总量与闭合类等效思考时间
            wtot = self._class_pool_wait(W)
            z_eff = base_think.copy()
            for c in net.closed_idx:
                z_eff[c] = base_think[c] + wtot[c]
            # 2) 裸站点网络 MVA（池等待只通过 z_eff 影响闭合类吞吐）
            sol = mva.solve(net, demands=D_base, approx=approx, cv=cv,
                            think_override=z_eff)
            X = sol.X
            # 3) 各池 Erlang-C 等待（持有时间含嵌套内层等待）
            new_W = self._pool_waits(X, W)
            # 松弛
            W = {p: self.relax * new_W[p] + (1 - self.relax) * prev_W[p]
                 for p in self.pool_names}
            w_err = max(abs(W[p] - prev_W[p]) / max(W[p], 1e-9)
                        for p in self.pool_names) if self.pool_names else 0.0
            x_err = (float(np.max(np.abs(X - prev_X)
                                  / np.maximum(np.abs(X), 1e-9)))
                     if prev_X is not None else np.inf)
            prev_W, prev_X = dict(W), X.copy()
            if w_err < self.tol and (prev_X is None or x_err < self.tol):
                converged = True
                break
        if not converged:
            notes.append("分层迭代达到上限未完全收敛，结果仅供趋势判断")

        # 最终组装
        wtot = self._class_pool_wait(W)
        z_eff = base_think.copy()
        for c in net.closed_idx:
            z_eff[c] = base_think[c] + wtot[c]
        final = mva.solve(net, demands=D_base, approx=approx, cv=cv,
                          think_override=z_eff)
        # 端到端响应时间 = 站点驻留 + 池等待
        R_total = final.R + wtot

        loads, utils, waits, queues = [], [], [], []
        for k, pname in enumerate(self.pool_names):
            v = self.acq_visits[pname]
            lam_p = float(np.sum(final.X * v))
            H_mean = self._mean_holding(pname, final.X, W)
            a_p = lam_p * H_mean
            cap = int(self.capacities[k])
            loads.append(a_p)
            utils.append(a_p / cap)
            waits.append(W[pname] if lam_p > 0 else 0.0)
            queues.append(lam_p * W[pname] if lam_p > 0 else 0.0)
            if a_p >= cap:
                notes.append(f"资源池 {pname} 已饱和（负荷 {a_p:.2f} ≥ "
                             f"容量 {cap}）")

        return LayeredSolution(
            base=_SolutionWithRT(final, R_total),
            pool_names=list(self.pool_names),
            pool_servers=self.capacities.copy(),
            pool_load=np.array(loads), pool_util=np.array(utils),
            pool_wait=np.array(waits), pool_queue=np.array(queues),
            original_S=net.S, iterations=it,
            converged=converged, notes=notes)

    # ------------------------------------------------------------------ #
    def _class_pool_wait(self, W: Dict[str, float]) -> np.ndarray:
        """各类每轮/每到达在所有资源池上的等待总量 [C]。"""
        net = self.net
        wtot = np.zeros(net.C)
        for pname in self.pool_names:
            wtot += self.acq_visits[pname] * W[pname]
        return wtot

    def _mean_holding(self, pname: str, X: np.ndarray,
                      W: Dict[str, float]) -> float:
        """池 pname 的申请率加权平均有效持有时间（含嵌套内层等待）。"""
        v = self.acq_visits[pname]
        H = self._bare_hold[pname].copy()
        for qname, arr in self._inner_refs[pname].items():
            per_outer = np.divide(arr, v, out=arr.copy(), where=v > 0)
            H = H + per_outer * W[qname]
        weights = X * v
        total = float(np.sum(weights))
        if total <= 0:
            return 0.0
        return float(np.sum(weights * H) / total)

    def _pool_waits(self, X: np.ndarray,
                    W: Dict[str, float]) -> Dict[str, float]:
        """由当前吞吐与持有时间，用 Erlang-C 计算各池每次申请等待。"""
        net = self.net
        out: Dict[str, float] = {}
        for pname in self.pool_names:
            v = self.acq_visits[pname]
            lam_p = float(np.sum(X * v))
            if lam_p <= 0:
                out[pname] = 0.0
                continue
            H_mean = self._mean_holding(pname, X, W)
            a_p = lam_p * H_mean
            cap = int(next(p.capacity for p in net.pools
                           if p.name == pname))
            if a_p >= cap:
                w = H_mean * (a_p - cap + 1.0) / cap
            else:
                f, _, _ = mva.queue_factor(a_p, cap)
                w = H_mean * (f - 1.0)
            out[pname] = max(0.0, w)
        return out
