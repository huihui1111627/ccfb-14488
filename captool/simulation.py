# -*- coding: utf-8 -*-
"""离散事件仿真（DES）——解析模型的验收基准。

覆盖能力
========
- 多类请求、显式路由、闭合用户（思考时间）与开放到达（泊松/MMPP/时变）；
- 三种排队规则：FCFS 多服务台、处理器共享 PS（份额变化时重算完成时刻，
  不做时间片近似）、IS 纯延迟；
- 资源池信号量：FIFO 等待队列，持有外层令牌阻塞内层，真实复现
  “拿着应用线程等数据库连接”的嵌套占用；
- 批均值法输出 95% 置信区间，支持多随机种子复现实验。

事件全部放在一个堆上：到达、服务完成、思考完成、池授权、MMPP 状态切换。
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np

from .network import QueueNetwork, FCFS, PS, IS, CLOSED
from .burst import MMPP2


@dataclass
class SimResult:
    R: np.ndarray                     # 各类平均响应时间（秒）
    X: np.ndarray                     # 各类吞吐（个/秒）
    util: np.ndarray                  # 各站点利用率（IS 为平均在场数）
    L: np.ndarray                     # 各站点平均在场请求数
    Lq: np.ndarray                    # 各站点平均排队请求数
    R_station: np.ndarray             # [c,s] 各类在各站平均驻留时间（秒）
    R_ci: np.ndarray                  # 各类响应时间 95% 置信半宽
    X_ci: np.ndarray
    util_ci: np.ndarray
    pool_util: np.ndarray             # 各资源池利用率
    pool_wait: np.ndarray             # 每次申请的平均池等待（秒）
    pool_queue: np.ndarray            # 池平均等待队列长度
    pool_names: List[str]
    seeds: List[int]
    n_completed: int
    sim_seconds: float
    warmup_seconds: float

    def summary(self, net: QueueNetwork) -> str:
        lines = [f"=== 仿真结果（seeds={self.seeds}, "
                 f"完成 {self.n_completed} 个请求, "
                 f"仿真时长 {self.sim_seconds:.0f}s）==="]
        lines.append(f"{'类':<10}{'RT(s)':>10}{'±95%':>9}"
                     f"{'吞吐':>10}{'±95%':>9}")
        for ci, c in enumerate(net.classes):
            lines.append(f"{c.name:<10}{self.R[ci]:10.4f}"
                         f"{self.R_ci[ci]:9.4f}{self.X[ci]:10.4f}"
                         f"{self.X_ci[ci]:9.4f}")
        lines.append(f"{'站点':<12}{'利用率':>9}{'±95%':>9}"
                     f"{'队长':>8}{'排队':>8}")
        for si, st in enumerate(net.stations):
            lines.append(f"{st.name:<12}{self.util[si]:9.4f}"
                         f"{self.util_ci[si]:9.4f}{self.L[si]:8.3f}"
                         f"{self.Lq[si]:8.3f}")
        if self.pool_names:
            lines.append(f"{'资源池':<12}{'利用率':>9}{'等待(s)':>10}"
                         f"{'等待队列':>10}")
            for k, pname in enumerate(self.pool_names):
                lines.append(f"{pname:<12}{self.pool_util[k]:9.4f}"
                             f"{self.pool_wait[k]:10.4f}"
                             f"{self.pool_queue[k]:10.3f}")
        return "\n".join(lines)


class _Job:
    __slots__ = ("jid", "cls", "step_idx", "arrival_time", "hold",
                 "station_in", "service_left", "wait_since",
                 "station_in_since", "ps_ver")

    def __init__(self, jid: int, cls: int, arrival: float):
        self.jid = jid
        self.cls = cls
        self.step_idx = 0
        self.arrival_time = arrival
        self.hold: List[str] = []
        self.station_in: Optional[int] = None
        self.service_left = 0.0
        self.wait_since = 0.0
        self.station_in_since = 0.0
        self.ps_ver = -1


class Simulator:
    def __init__(self, net: QueueNetwork, seed: int = 1):
        self.net = net
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self._reset()

    def _reset(self):
        net = self.net
        self.eq: List[tuple] = []
        self.now = 0.0
        self._seq = 0
        self.jobs: Dict[int, _Job] = {}
        self._next_jid = 0

        self.fcfs_queue: List[List[int]] = [[] for _ in net.stations]
        self.fcfs_busy = np.zeros(net.S, dtype=int)
        self.ps_members: List[set] = [set() for _ in net.stations]
        self.ps_ver = np.zeros(net.S, dtype=int)        # 站点级事件版本计数器
        self.ps_rate = np.zeros(net.S)                  # 每个成员当前服务速率
        self.ps_last = np.zeros(net.S)                  # 本轮份额记账起点
        self.is_present: List[set] = [set() for _ in net.stations]

        self.pool_free = {p.name: p.capacity for p in net.pools}
        self.pool_waiters: Dict[str, List[int]] = {
            p.name: [] for p in net.pools}

        # 时间积分
        self._last_sample = 0.0
        # 每类每站驻留时间累计（入站到出站）与完成次数
        self.station_time = np.zeros((net.C, net.S))
        self.station_visits = np.zeros((net.C, net.S))
        self.pool_grant_count = {p.name: 0 for p in net.pools}
        self.pool_wait_sum = {p.name: 0.0 for p in net.pools}
        self.pool_area_held = {p.name: 0.0 for p in net.pools}
        self.pool_area_wait = {p.name: 0.0 for p in net.pools}

        # 批均值
        self.n_batches = 5
        self.batch_rt: Dict[int, List[List[float]]] = {
            ci: [] for ci in range(net.C)}
        self.batch_x: Dict[int, List[int]] = {
            ci: [] for ci in range(net.C)}
        self.batch_busy: List[np.ndarray] = []
        self.batch_L: List[np.ndarray] = []
        self.batch_Lq: List[np.ndarray] = []
        self._cur_rt: Dict[int, List[float]] = {
            ci: [] for ci in range(net.C)}
        self._cur_x = np.zeros(net.C, dtype=int)
        self._cur_busy = np.zeros(net.S)
        self._cur_L = np.zeros(net.S)
        self._cur_Lq = np.zeros(net.S)
        self.completed = 0
        self.warmup = 0.0
        self.warmup_done = False

    # ------------------------------------------------------------------ #
    def _push(self, t: float, kind: str, jid: int, data=None):
        heapq.heappush(self.eq, (max(t, self.now), self._seq,
                                 kind, jid, data))
        self._seq += 1

    def _pop(self):
        return heapq.heappop(self.eq)

    # ------------------------------------------------------------------ #
    # 时间积分（队长/利用率/池占用）
    # ------------------------------------------------------------------ #
    def _integrate(self):
        dt = self.now - self._last_sample
        if dt <= 0:
            return
        for si, st in enumerate(self.net.stations):
            if st.discipline == IS:
                self._cur_L[si] += len(self.is_present[si]) * dt
            elif st.discipline == FCFS:
                n = self.fcfs_busy[si] + len(self.fcfs_queue[si])
                self._cur_L[si] += n * dt
                self._cur_Lq[si] += len(self.fcfs_queue[si]) * dt
                self._cur_busy[si] += self.fcfs_busy[si] * dt
            else:
                n = len(self.ps_members[si])
                self._cur_L[si] += n * dt
                self._cur_busy[si] += min(n, st.servers) * dt
                self._cur_Lq[si] += max(0, n - st.servers) * dt
        for p in self.net.pools:
            held = (p.capacity - self.pool_free[p.name]
                    - len(self.pool_waiters[p.name]))
            self.pool_area_held[p.name] += max(0, held) * dt
            self.pool_area_wait[p.name] += len(
                self.pool_waiters[p.name]) * dt
        self._last_sample = self.now

    # ------------------------------------------------------------------ #
    # 作业流转
    # ------------------------------------------------------------------ #
    def _new_job(self, cls: int):
        jid = self._next_jid
        self._next_jid += 1
        self.jobs[jid] = _Job(jid, cls, self.now)
        self._advance(self.jobs[jid])

    def _advance(self, job: _Job):
        net = self.net
        c = net.classes[job.cls]
        if job.step_idx >= len(c.route):
            self._finish(job)
            return
        step = c.route[job.step_idx]
        if step.acquire:
            if self.pool_free[step.acquire] > 0:
                self.pool_free[step.acquire] -= 1
                self.pool_grant_count[step.acquire] += 1
                job.hold.append(step.acquire)
            else:
                # 池满阻塞：继续持有已获得的外层令牌
                job.wait_since = self.now
                self.pool_waiters[step.acquire].append(job.jid)
                return
        self._enter_station(job, net.si(step.station), step.service)

    def _enter_station(self, job: _Job, si: int, service: float):
        st = self.net.stations[si]
        job.station_in = si
        job.station_in_since = self.now
        job.service_left = (self.rng.exponential(service)
                            if service > 0 else 0.0)
        if st.discipline == FCFS:
            self.fcfs_queue[si].append(job.jid)
            self._schedule_fcfs(si)
        elif st.discipline == PS:
            # 关键：必须在新作业加入成员集 **之前** 按旧份额结算在场
            # 成员，否则新作业会被旧速率错误扣减一段它并未在场的工作量。
            if self.ps_members[si]:
                self._ps_account(si)
            else:
                self.ps_last[si] = self.now
                self.ps_rate[si] = 0.0
            self.ps_members[si].add(job.jid)
            # 此处直接重排（不再二次结算）：记账时刻已锚定到 now
            self.ps_ver[si] += 1
            n = len(self.ps_members[si])
            rate = min(st.servers, n) / n
            self.ps_rate[si] = rate
            ver = self.ps_ver[si]
            for jid in self.ps_members[si]:
                j = self.jobs[jid]
                j.ps_ver = ver
                self._push(self.now + j.service_left / rate,
                           "ps_done", jid, (si, ver))
        else:
            self.is_present[si].add(job.jid)
            self._push(self.now + job.service_left, "is_done",
                       job.jid, si)

    def _leave_station(self, job: _Job):
        net = self.net
        si = job.station_in
        st = net.stations[si]
        c = net.classes[job.cls]
        step = c.route[job.step_idx]
        # 驻留时间统计（预热期不计入）
        if self.warmup_done:
            self.station_time[job.cls, si] += self.now - job.station_in_since
            self.station_visits[job.cls, si] += 1
        if st.discipline == IS:
            self.is_present[si].discard(job.jid)
        job.station_in = None
        job.service_left = 0.0
        if step.release:
            if step.release not in job.hold:
                raise RuntimeError("内部错误：释放未持有的令牌")
            job.hold.remove(step.release)
            self.pool_free[step.release] += 1
            self._grant_pool(step.release)
        job.step_idx += 1
        self._advance(job)

    def _grant_pool(self, pname: str):
        """池出现空闲令牌时，把授权动作作为零延迟事件投递到事件堆。

        不在释放调用内直接唤醒等待者：被唤醒作业进站（尤其 PS 站）可能
        连锁触发完成与再次释放，递归重入会在 while 循环中途改写 pool_free
        和等待队列，造成令牌计数泄漏。事件化后所有状态变更都在主循环的
        单一步骤内串行完成。
        """
        if self.pool_waiters[pname] and self.pool_free[pname] > 0:
            self._push(self.now, "pool_grant", -1, pname)

    # ------------------------------------------------------------------ #
    # FCFS
    # ------------------------------------------------------------------ #
    def _drain_pool(self, pname: str):
        """在主循环中串行唤醒等待某池的 FIFO 队首作业。"""
        q = self.pool_waiters[pname]
        while q and self.pool_free[pname] > 0:
            jid = q.pop(0)
            job = self.jobs.get(jid)
            if job is None:
                continue
            if self.warmup_done:
                self.pool_wait_sum[pname] += self.now - job.wait_since
            self.pool_free[pname] -= 1
            self.pool_grant_count[pname] += 1
            job.hold.append(pname)
            c = self.net.classes[job.cls]
            step = c.route[job.step_idx]
            self._enter_station(job, self.net.si(step.station),
                                step.service)

    def _schedule_fcfs(self, si: int):
        st = self.net.stations[si]
        while self.fcfs_busy[si] < st.servers and self.fcfs_queue[si]:
            jid = self.fcfs_queue[si].pop(0)
            job = self.jobs[jid]
            self.fcfs_busy[si] += 1
            self._push(self.now + job.service_left, "fcfs_done",
                       jid, si)

    # ------------------------------------------------------------------ #
    # PS：成员变化 → 按经过时间扣减剩余工作量 → 按新份额重排
    # ------------------------------------------------------------------ #
    def _ps_account(self, si: int):
        """按上一份额区间的速率扣减各成员的剩余工作量，并推进记账时刻。

        必须在改变成员集合/速率 **之前** 调用：用旧速率 ``ps_rate[si]``
        结算 ``now - ps_last[si]`` 这段区间，然后把记账时刻更新到 now。
        """
        members = self.ps_members[si]
        dt = self.now - self.ps_last[si]
        if dt > 0 and members:
            consumed = self.ps_rate[si] * dt
            for jid in members:
                job = self.jobs[jid]
                job.service_left = max(1e-12, job.service_left - consumed)
        self.ps_last[si] = self.now

    def _reschedule_ps(self, si: int):
        """成员变化 → 按旧速率结算旧区间 → 用新份额重排完成事件。

        处理器共享下每个成员获得速率 ``min(m,n)/n``。每次重排递增站点
        版本计数器并把新版本号写入每个成员的 ``job.ps_ver`` 及其完成
        事件；事件触发时与作业自身版本严格匹配，旧事件天然失效。
        """
        members = self.ps_members[si]
        self._ps_account(si)              # 先用旧速率结算到当前时刻
        self.ps_ver[si] += 1
        if not members:
            self.ps_rate[si] = 0.0
            return
        rate = min(self.net.stations[si].servers, len(members)) / len(members)
        self.ps_rate[si] = rate
        ver = self.ps_ver[si]
        for jid in members:
            job = self.jobs[jid]
            job.ps_ver = ver
            self._push(self.now + job.service_left / rate,
                       "ps_done", jid, (si, ver))

    # ------------------------------------------------------------------ #
    def _finish(self, job: _Job):
        if self.warmup_done:
            rt = self.now - job.arrival_time
            self._cur_rt[job.cls].append(rt)
            self._cur_x[job.cls] += 1
            self.completed += 1
        ci = job.cls
        del self.jobs[job.jid]
        c = self.net.classes[ci]
        if c.kind == CLOSED:
            z = self.rng.exponential(c.think) if c.think > 0 else 0.0
            self._push(self.now + z, "think_done", ci, None)

    # ------------------------------------------------------------------ #
    # 开放到达
    # ------------------------------------------------------------------ #
    def _arm_open(self, ci: int, mmpp: Optional[MMPP2],
                  rate_fn, lam_hi: float):
        c = self.net.classes[ci]
        if rate_fn is not None:
            self._push(self.now + self.rng.exponential(1.0 / lam_hi),
                       "open_nhpp", ci, (rate_fn, lam_hi))
            return
        if mmpp is not None:
            state = getattr(self, "_mmpp_state", {}).get(ci, 0)
            lam = mmpp.lambda1 if state == 1 else mmpp.lambda0
        else:
            lam = c.arrival
        self._push(self.now + self.rng.exponential(1.0 / lam),
                   "open_arrival", ci, mmpp)

    # ------------------------------------------------------------------ #
    # 主循环
    # ------------------------------------------------------------------ #
    def run(self,
            min_completed: int = 60000,
            max_seconds: float = 600.0,
            warmup_frac: float = 0.2,
            n_batches: int = 5,
            mmpp: Optional[MMPP2] = None,
            rate_fn: Optional[Callable[[float], float]] = None) -> SimResult:
        net = self.net
        self.n_batches = n_batches
        self.warmup = max_seconds * warmup_frac
        measure_len = max_seconds - self.warmup
        batch_len = measure_len / n_batches

        for ci in net.closed_idx:
            c = net.classes[ci]
            for _ in range(c.population):
                self._push(0.0, "think_done", ci, None)
        self._mmpp_state: Dict[int, int] = {}
        lam_hi = None
        if rate_fn is not None:
            grid = np.linspace(0, max_seconds, 400)
            lam_hi = float(max(rate_fn(t) for t in grid)) * 1.05 + 1e-9
        for ci in net.open_idx:
            if mmpp is not None:
                self._mmpp_state[ci] = 0
                self._push(self.rng.exponential(1.0 / mmpp.r01),
                           "mmpp_switch", ci, mmpp)
            self._arm_open(ci, mmpp, rate_fn, lam_hi or 0.0)

        next_batch = self.warmup + batch_len
        batch_no = 0
        stop_early = False

        while self.eq:
            t, _, kind, jid, data = self._pop()
            if t >= max_seconds:
                self.now = max_seconds
                self._integrate()
                break
            self.now = t
            self._integrate()

            if kind == "think_done":
                self._new_job(jid)
            elif kind == "open_arrival":
                self._new_job(jid)
                self._arm_open(jid, data, None, 0.0)
            elif kind == "mmpp_switch":
                m: MMPP2 = data
                self._mmpp_state[jid] = 1 - self._mmpp_state[jid]
                rate = (m.r10 if self._mmpp_state[jid] == 1
                        else m.r01)
                self._push(self.now + self.rng.exponential(1.0 / rate),
                           "mmpp_switch", jid, m)
            elif kind == "open_nhpp":
                fn, hi = data
                if self.rng.random() < fn(self.now) / hi:
                    self._new_job(jid)
                self._arm_open(jid, None, fn, hi)
            elif kind == "pool_grant":
                pname = data
                self._drain_pool(pname)
            elif kind == "fcfs_done":
                si = data
                self.fcfs_busy[si] -= 1
                self._leave_station(self.jobs[jid])
                self._schedule_fcfs(si)
            elif kind == "ps_done":
                si, ver = data
                job = self.jobs.get(jid)
                if (job is None or job.station_in != si
                        or ver != job.ps_ver):
                    continue                    # 该作业已被重排的过期占位事件
                # 先结算最后一份额区间的工作量，移出成员集后再触发离开
                self._ps_account(si)
                job.service_left = 0.0
                self.ps_members[si].discard(jid)
                self._leave_station(job)
                self._reschedule_ps(si)
            elif kind == "is_done":
                self._leave_station(self.jobs[jid])

            # 预热边界
            if not self.warmup_done and self.now >= self.warmup:
                self.warmup_done = True
                self._cur_rt = {ci: [] for ci in range(net.C)}
                self._cur_x[:] = 0
                self._cur_busy[:] = 0
                self._cur_L[:] = 0
                self._cur_Lq[:] = 0
            # 批切分
            if (self.warmup_done and self.now >= next_batch
                    and batch_no < n_batches):
                self._flush_batch(batch_len)
                batch_no += 1
                next_batch = self.warmup + (batch_no + 1) * batch_len
            # 提前停止：完成数达标且已过预热
            if (min_completed > 0 and self.completed >= min_completed
                    and self.warmup_done and batch_no >= 2):
                stop_early = True
                break

        if stop_early or batch_no < n_batches:
            tail = max(0.0, self.now - self.warmup - batch_no * batch_len)
            if tail > 0:
                self._flush_batch(tail)
        return self._build_result()

    def _flush_batch(self, length: float):
        if length <= 0:
            return
        net = self.net
        for ci in range(net.C):
            self.batch_rt[ci].append(list(self._cur_rt[ci]))
            self.batch_x[ci].append(int(self._cur_x[ci]))
        self.batch_busy.append((self._cur_busy / length).copy())
        self.batch_L.append((self._cur_L / length).copy())
        self.batch_Lq.append((self._cur_Lq / length).copy())
        self._cur_rt = {ci: [] for ci in range(net.C)}
        self._cur_x[:] = 0
        self._cur_busy[:] = 0
        self._cur_L[:] = 0
        self._cur_Lq[:] = 0

    # ------------------------------------------------------------------ #
    @staticmethod
    def _half_width(samples) -> float:
        if len(samples) < 2:
            return 0.0
        arr = np.asarray(samples, dtype=float)
        t_table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776,
                   5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}
        t = t_table.get(len(arr) - 1, 1.96)
        return float(t * np.std(arr, ddof=1) / math.sqrt(len(arr)))

    def _build_result(self) -> SimResult:
        net = self.net
        total_t = max(self.now - self.warmup, 1e-9)

        R, R_ci, X, X_ci = np.zeros(net.C), np.zeros(net.C), \
            np.zeros(net.C), np.zeros(net.C)
        for ci in range(net.C):
            all_rt = [v for b in self.batch_rt[ci] for v in b]
            R[ci] = float(np.mean(all_rt)) if all_rt else 0.0
            R_ci[ci] = self._half_width(
                [np.mean(b) for b in self.batch_rt[ci] if b])
            X[ci] = sum(self.batch_x[ci]) / total_t
            n_b = max(len(self.batch_x[ci]), 1)
            X_ci[ci] = self._half_width(
                [n / (total_t / n_b) for n in self.batch_x[ci]])

        util, util_ci, L, Lq = (np.zeros(net.S), np.zeros(net.S),
                                np.zeros(net.S), np.zeros(net.S))
        for si, st in enumerate(net.stations):
            if st.discipline == IS:
                lb = [b[si] for b in self.batch_L]
                L[si] = float(np.mean(lb)) if lb else 0.0
                util[si] = L[si]
                util_ci[si] = self._half_width(lb)
            else:
                ub = [b[si] / st.servers for b in self.batch_busy]
                util[si] = float(np.mean(ub)) if ub else 0.0
                util_ci[si] = self._half_width(ub)
                L[si] = (float(np.mean([b[si] for b in self.batch_L]))
                         if self.batch_L else 0.0)
                Lq[si] = (float(np.mean([b[si] for b in self.batch_Lq]))
                          if self.batch_Lq else 0.0)

        R_station = np.divide(
            self.station_time, self.station_visits,
            out=np.zeros_like(self.station_time),
            where=self.station_visits > 0)

        pool_names = [p.name for p in net.pools]
        pool_util = np.zeros(len(pool_names))
        pool_wait = np.zeros(len(pool_names))
        pool_queue = np.zeros(len(pool_names))
        for k, pname in enumerate(pool_names):
            cap = next(p.capacity for p in net.pools if p.name == pname)
            pool_util[k] = self.pool_area_held[pname] / (cap * self.now)
            g = self.pool_grant_count[pname]
            pool_wait[k] = self.pool_wait_sum[pname] / g if g else 0.0
            pool_queue[k] = self.pool_area_wait[pname] / self.now

        return SimResult(
            R=R, X=X, util=util, L=L, Lq=Lq, R_station=R_station,
            R_ci=R_ci, X_ci=X_ci, util_ci=util_ci,
            pool_util=pool_util, pool_wait=pool_wait,
            pool_queue=pool_queue, pool_names=pool_names,
            seeds=[self.seed], n_completed=self.completed,
            sim_seconds=self.now, warmup_seconds=self.warmup)


def simulate(net: QueueNetwork, seed: int = 1, **kwargs) -> SimResult:
    """单次仿真便捷入口。"""
    return Simulator(net, seed=seed).run(**kwargs)


def simulate_replicates(net: QueueNetwork, n_rep: int = 4,
                        base_seed: int = 100, **kwargs) -> SimResult:
    """多种子重复仿真，均值 + 重复间 95% 置信区间。"""
    results = [simulate(net, seed=base_seed + i, **kwargs)
               for i in range(n_rep)]
    ref = results[0]
    mR = np.array([r.R for r in results])
    mX = np.array([r.X for r in results])
    mU = np.array([r.util for r in results])
    ref.R, ref.X, ref.util = mR.mean(0), mX.mean(0), mU.mean(0)
    if n_rep > 1:
        f = 1.96 / math.sqrt(n_rep)
        ref.R_ci = f * mR.std(axis=0, ddof=1)
        ref.X_ci = f * mX.std(axis=0, ddof=1)
        ref.util_ci = f * mU.std(axis=0, ddof=1)
    ref.L = np.mean([r.L for r in results], axis=0)
    ref.Lq = np.mean([r.Lq for r in results], axis=0)
    ref.R_station = np.mean([r.R_station for r in results], axis=0)
    ref.pool_util = np.mean([r.pool_util for r in results], axis=0)
    ref.pool_wait = np.mean([r.pool_wait for r in results], axis=0)
    ref.pool_queue = np.mean([r.pool_queue for r in results], axis=0)
    ref.seeds = [r.seeds[0] for r in results]
    ref.n_completed = int(np.mean([r.n_completed for r in results]))
    return ref
