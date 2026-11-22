"""离散事件仿真 (DES) —— 被评估系统的替代环境。

支持的真实细节:

* 三种调度: FCFS 先到先服务 / PS 处理器共享(按剩余服务量精确驱动,
  支持任意变异系数) / IS 纯延迟(无限服务台);
* 有限资源池: 请求进入带池站点先抢池令牌(连接/线程), 抢不到则
  **持父站点资源阻塞等待**, 精确复现嵌套占用;
* 同步嵌套调用: 父站点自身服务结束后释放服务台、但继续持有池令牌,
  子调用全部返回后才释放令牌并出队;
* 闭合类(固定用户数 + think time)与开放类(齐次/非齐次泊松到达);
* 服务时间可用指数 / 定长 / Gamma(任意变异系数 cv)。
"""

from __future__ import annotations

import copy
import heapq
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .network import (Network, ServiceClass, Station, RouteStep,
                      SCHED_FCFS, SCHED_PS, SCHED_IS)

# 事件类型
_E_ARRIVAL = 1        # 请求(含子调用)到达站点
_E_SVC_DONE = 2       # 站点自身服务完成
_E_THINK_DONE = 3     # 闭合用户 think 结束
_E_WARM = 4           # warmup 结束, 冻结统计基线
_E_END = 5            # 仿真结束
_E_OPEN_ARRIVAL = 6   # 开放类外部到达(独立到达链节拍)


@dataclass
class _Frame:
    """一次站点访问的执行现场。

    帧在整个站点访问期间留在 ``req.frames`` 栈中(含子调用期间),
    只有访问完全结束才弹出。``pending`` 是该帧自身服务结束后
    仍待发起的同步子调用 (callee 站名) 队列, 子调用返回后由父帧
    逐次消费——这保证同一子站点被调用多次时每次都有独立的帧。
    """
    station: str
    demand: float
    pending: List[str] = field(default_factory=list)
    entry_time: float = 0.0
    pool_wait_start: Optional[float] = None


@dataclass
class _Request:
    rid: int
    cls: ServiceClass
    birth: float
    frames: List[_Frame] = field(default_factory=list)
    in_think: bool = False          # 闭合用户 think 期间不在任何站点


@dataclass
class _Snapshot:
    """可在 warmup 时刻清零的累计量。"""
    arrivals: int = 0
    visits_completed: int = 0
    busy_area: float = 0.0           # 忙服务台数对时间积分
    queue_area: float = 0.0          # 服务队列长度积分(FCFS 等待 + PS 等待)
    pool_wait_area: float = 0.0      # 等池令牌的请求数积分
    in_system_area: float = 0.0      # 站内请求总数积分
    residence_sum: float = 0.0
    residence_cnt: int = 0
    pool_wait_time_sum: float = 0.0


class _StationState:
    def __init__(self, st: Station) -> None:
        self.st = st
        # FCFS
        self.busy = 0
        self.fcfs_q: deque = deque()
        # PS: 处理器共享, 所有在站作业均分总能力(无独立等待队列)
        self.ps_active: Dict[int, Tuple[_Request, float]] = {}  # rid -> (req, 剩余服务量)
        self._ps_last_t = 0.0
        self._ps_gen = 0
        # 资源池
        self.tokens = st.pool_capacity if st.pool_capacity is not None else None
        self.pool_q: deque = deque()
        # 面积积分
        self.snap = _Snapshot()
        self._q_len = 0
        self._pool_len = 0
        self._in_sys = 0
        self._t_last = 0.0
        self.warm_snap: Optional[_Snapshot] = None

    def charge(self, now: float) -> None:
        dt = now - self._t_last
        self.snap.busy_area += self.busy * dt
        self.snap.queue_area += self._q_len * dt
        self.snap.pool_wait_area += self._pool_len * dt
        self.snap.in_system_area += self._in_sys * dt
        self._t_last = now

    def mark_warm(self, now: float) -> None:
        self.charge(now)
        self.warm_snap = copy.deepcopy(self.snap)

    def delta(self) -> _Snapshot:
        w = self.warm_snap
        out = _Snapshot()
        for k in self.snap.__dataclass_fields__:  # type: ignore[attr-defined]
            setattr(out, k, getattr(self.snap, k) - getattr(w, k))
        return out


class Simulator:
    """单条离散事件仿真。"""

    def __init__(self, net: Network, seed: int = 0,
                 rate_overrides: Optional[Dict[str, Callable[[float], float]]] = None,
                 ) -> None:
        self.net = net
        self.rng = np.random.default_rng(seed)
        self.rate_overrides = rate_overrides or {}
        self.st: Dict[str, _StationState] = {
            name: _StationState(s) for name, s in net.stations.items()}
        self._heap: List[Tuple[float, int, int, object]] = []
        self._seq = 0
        self._rid = 0
        self.now = 0.0
        self._end_time = 0.0
        self._warmup = 0.0
        self._horizon = 0.0
        self._cls_resp: Dict[str, List[float]] = {c: [] for c in net.classes}
        self._cls_count: Dict[str, int] = {c: 0 for c in net.classes}
        self._cls_warm_count: Dict[str, int] = {c: 0 for c in net.classes}
        self._cls_warm_sum: Dict[str, float] = {c: 0.0 for c in net.classes}
        # 测量窗口内的根请求到达数(开放类=外部到达, 闭合类=think 结束)
        self._root_arrivals: Dict[str, int] = {c: 0 for c in net.classes}
        self._root_warm: Dict[str, int] = {c: 0 for c in net.classes}

    # ------------------------------------------------------------------ #
    # 随机量
    # ------------------------------------------------------------------ #
    def _service_time(self, mean: float, cv: float) -> float:
        if mean <= 0:
            return 0.0
        if cv <= 1e-9:
            return mean
        if abs(cv - 1.0) < 1e-9:
            return self.rng.exponential(mean)
        shape = 1.0 / (cv * cv)
        return self.rng.gamma(shape, mean / shape)

    def _think_time(self, mean: float) -> float:
        return self.rng.exponential(mean) if mean > 0 else 0.0

    # ------------------------------------------------------------------ #
    # 事件堆
    # ------------------------------------------------------------------ #
    def _push(self, t: float, etype: int, payload: object) -> None:
        self._seq += 1
        heapq.heappush(self._heap, (max(t, self.now), self._seq, etype, payload))

    def run(self, warmup: float, horizon: float) -> "Simulator":
        self._warmup = warmup
        self._horizon = horizon
        self._end_time = warmup + horizon
        for cname, cls in self.net.classes.items():
            if cls.kind == "closed":
                for _ in range(cls.population):
                    req = self._new_request(cls, 0.0)
                    stagger = self.rng.uniform(0, min(1.0, warmup / 10.0))
                    self._push(stagger, _E_ARRIVAL,
                               (req, cls.route[0].station))
            else:
                # 开放类: 独立自调度到达链, 与请求完成解耦
                self._schedule_open_arrival(cname, cls, 0.0)
        self._push(warmup, _E_WARM, None)
        self._push(self._end_time, _E_END, None)

        while self._heap:
            now, _, etype, payload = heapq.heappop(self._heap)
            self.now = now
            if etype == _E_WARM:
                self._mark_warmup()
                continue
            if etype == _E_END:
                break
            if etype == _E_OPEN_ARRIVAL:
                cname = payload  # type: ignore[assignment]
                cls = self.net.classes[cname]
                req = self._new_request(cls, self.now)
                # 自我延续下一个外部到达(独立节拍)
                self._schedule_open_arrival(cname, cls, self.now)
                self._push(self.now, _E_ARRIVAL,
                           (req, cls.route[0].station))
                continue
            if etype == _E_ARRIVAL:
                req, station = payload  # type: ignore[misc]
                self._on_arrival(req, station)
            elif etype == _E_SVC_DONE:
                req, station, gen = payload  # type: ignore[misc]
                self._on_service_done(req, station, gen)
            elif etype == _E_THINK_DONE:
                req = payload  # type: ignore[assignment]
                req.in_think = False
                # 根到达计数统一在 _on_arrival 里做, 这里重置出生时刻
                req.birth = self.now
                self._push(self.now, _E_ARRIVAL,
                           (req, req.cls.route[0].station))
        for ss in self.st.values():
            ss.charge(self._end_time)
        return self

    def _new_request(self, cls: ServiceClass, birth: float) -> _Request:
        self._rid += 1
        return _Request(rid=self._rid, cls=cls, birth=birth)

    # ------------------------------------------------------------------ #
    # 开放类到达(齐次泊松 / 非齐次泊松, 稀疏法)
    # ------------------------------------------------------------------ #
    def _open_rate(self, cname: str, t: float, base: float) -> float:
        f = self.rate_overrides.get(cname)
        return f(t) if f is not None else base

    def _schedule_open_arrival(self, cname: str, cls: ServiceClass,
                               from_t: float) -> None:
        """排下一个**外部**到达并自我延续到达链。

        到达链独立于请求生命周期(关键: 否则轻载时系统会退化成
        "完成一个才放一个"的闭合行为, 吞吐被响应时间拖死)。
        不预设截止, 靠 END 事件停止消费。
        """
        rate_fn = self.rate_overrides.get(cname)
        if rate_fn is None:
            nxt = from_t + self.rng.exponential(1.0 / cls.arrival_rate)
            self._push(nxt, _E_OPEN_ARRIVAL, cname)
            return
        # NHPP 稀疏法
        t = from_t
        while True:
            window = 30.0
            grid = np.linspace(t, t + window, 13)
            lam_max = max(rate_fn(x) for x in grid)
            if lam_max <= 0:
                t += window
                continue
            t = t + self.rng.exponential(1.0 / lam_max)
            if self.rng.random() <= rate_fn(t) / lam_max:
                self._push(t, _E_OPEN_ARRIVAL, cname)
                return

    # ------------------------------------------------------------------ #
    # 站点进入 / 资源池
    # ------------------------------------------------------------------ #
    def _on_arrival(self, req: _Request, station: str) -> None:
        ss = self.st[station]
        ss.charge(self.now)
        ss.snap.arrivals += 1
        ss._in_sys += 1
        if not req.frames:
            # 根到达: 开放类外部到达或闭合类 think 结束后的首站。
            # 计数按到达时刻归属测量窗口, warmup 基线在 _mark_warmup 冻结
            self._root_arrivals[req.cls.name] += 1
        # 根路由步用类定义; 子调用由父帧的调用队列驱动, 需求/深层调用
        # 仍取该类在该站点的路由步, 但生命周期由父帧管理
        step = self._step_for(req.cls, station)
        # 整数次 = 确定性重复调用; 非整数次 = 概率性调用(伯努利),
        # 例如 0.6 表示该站点在本次访问中有 60% 概率被调用一次
        pending: List[str] = []
        for call in step.calls:
            whole = int(math.floor(call.calls))
            pending.extend([call.callee] * whole)
            frac = call.calls - whole
            if frac > 0 and self.rng.random() < frac:
                pending.append(call.callee)
        frame = _Frame(station=station, demand=step.demand,
                       pending=pending, entry_time=self.now)
        req.frames.append(frame)
        if ss.tokens is not None:
            if ss.tokens > 0:
                ss.tokens -= 1
                self._enter_service(req, station)
            else:
                frame.pool_wait_start = self.now
                ss.pool_q.append(req)
                ss._pool_len += 1
        else:
            self._enter_service(req, station)

    @staticmethod
    def _step_for(cls: ServiceClass, station: str) -> RouteStep:
        """该类在指定站点的路由步(定义需求与更深层调用)。"""
        for st in cls.route:
            if st.station == station:
                return st
        raise KeyError(f"类 {cls.name} 没有站点 {station} 的路由步")

    def _enter_service(self, req: _Request, station: str) -> None:
        ss = self.st[station]
        st = ss.st
        frame = req.frames[-1]
        if st.sched == SCHED_IS:
            ss.busy += 1
            svc = self._service_time(frame.demand, req.cls.cv)
            self._push(self.now + svc, _E_SVC_DONE, (req, station, -1))
        elif st.sched == SCHED_FCFS:
            ss.fcfs_q.append(req)
            ss._q_len += 1
            self._dispatch_fcfs(station)
        else:  # PS: 处理器共享, 所有在站作业均分 m 台总服务能力
            total = self._service_time(frame.demand, req.cls.cv)
            self._ps_join(station, req, total)

    def _dispatch_fcfs(self, station: str) -> None:
        ss = self.st[station]
        st = ss.st
        while ss.busy < (st.servers or 1) and ss.fcfs_q:
            req = ss.fcfs_q.popleft()
            ss._q_len -= 1
            ss.busy += 1
            svc = self._service_time(req.frames[-1].demand, req.cls.cv)
            self._push(self.now + svc, _E_SVC_DONE, (req, station, -1))

    # ------------------------------------------------------------------ #
    # PS: m 台总能力, n 个在站作业各得 m/n 速率;
    # 任意状态变更按剩余服务量重算下一完成事件。
    # busy = min(n, m): 忙面积口径与 FCFS 一致, 利用率可直接对比。
    # ------------------------------------------------------------------ #
    def _ps_advance(self, ss: _StationState) -> None:
        """到达等非完成事件: 按经过时间扣减所有在站作业剩余量, 不移除。

        PS 每作业速率 = min(1, m/n): n<=m 时各独占一台(速率1),
        n>m 时 m 台能力在 n 个作业间均分。总能力恒为 min(n,m),
        与 busy=min(n,m) 的记账一致。
        """
        dt = self.now - ss._ps_last_t
        n = len(ss.ps_active)
        m = ss.st.servers or 1
        if n > 0 and dt > 0:
            share = dt * min(1.0, m / n)
            ss.ps_active = {
                rid: (r, max(rem - share, 0.0))
                for rid, (r, rem) in ss.ps_active.items()}
        ss._ps_last_t = self.now

    def _ps_join(self, station: str, req: _Request, work: float) -> None:
        ss = self.st[station]
        self._ps_advance(ss)
        ss.ps_active[req.rid] = (req, work)
        ss.busy = min(len(ss.ps_active), ss.st.servers or 1)
        self._ps_reschedule(station)

    def _ps_settle(self, ss: _StationState) -> Optional[_Request]:
        """服务完成时刻: 精确移除一个剩余量最少的作业。

        PS 完成事件总在"某作业恰好做完"的时刻触发, 故不做时间摊薄
        (浮点摊薄在长仿真里会吞掉作业、造成闭合用户永久流失),
        其余作业按本区间份额扣减, 完成者精确弹出。
        """
        n = len(ss.ps_active)
        m = ss.st.servers or 1
        dt = self.now - ss._ps_last_t
        share = dt * min(1.0, m / n)
        winner_rid = min(ss.ps_active,
                         key=lambda rid: ss.ps_active[rid][1])
        winner_req, _ = ss.ps_active.pop(winner_rid)
        ss.ps_active = {
            rid: (r, max(rem - share, 0.0))
            for rid, (r, rem) in ss.ps_active.items()}
        ss._ps_last_t = self.now
        return winner_req

    def _ps_reschedule(self, station: str) -> None:
        ss = self.st[station]
        ss._ps_gen += 1
        gen = ss._ps_gen
        ss._ps_last_t = self.now
        if not ss.ps_active:
            ss.busy = 0
            return
        n = len(ss.ps_active)
        m = ss.st.servers or 1
        ss.busy = min(n, m)
        # 下一完成: 最小剩余需求 / 每作业速率 min(1, m/n)
        min_rem = min(rem for _, rem in ss.ps_active.values())
        rate_each = min(1.0, m / n)
        self._push(self.now + min_rem / rate_each,
                   _E_SVC_DONE, (None, station, gen))

    # ------------------------------------------------------------------ #
    # 服务完成 -> 子调用 -> 释放
    # ------------------------------------------------------------------ #
    def _on_service_done(self, req: Optional[_Request], station: str,
                         gen: int) -> None:
        ss = self.st[station]
        st = ss.st
        ss.charge(self.now)

        if st.sched == SCHED_PS:
            if gen != ss._ps_gen:
                return
            winner = self._ps_settle(ss)
            if winner is None:
                return
            req = winner
            self._ps_reschedule(station)
        else:
            ss.busy -= 1
            if st.sched == SCHED_FCFS:
                self._dispatch_fcfs(station)

        # 自身服务结束: 进入子调用阶段(服务台已释放, 池令牌仍持有)
        self._advance(req, station)  # type: ignore[arg-type]

    def _advance(self, req: _Request, station: str) -> None:
        """帧状态机: 调用队列非空则发起下一个子调用, 否则退出本站点。"""
        frame = req.frames[-1]
        assert frame.station == station
        if frame.pending:
            callee = frame.pending.pop(0)
            # 父帧保留在栈中, 子帧入栈
            self._push(self.now, _E_ARRIVAL, (req, callee))
            return
        self._exit_station(req, station)

    def _exit_station(self, req: _Request, station: str) -> None:
        """站点访问结束: 弹帧、还令牌, 然后决定下一步。

        三种去向:
        1. 仍在嵌套调用中(栈非空): 回到父帧继续其子调用队列;
        2. 嵌套已结束但还有平铺路由步: 进入下一个根级路由步;
        3. 路由走完: 整条请求完成。
        """
        ss = self.st[station]
        ss.charge(self.now)
        frame = req.frames.pop()
        assert frame.station == station
        ss.snap.visits_completed += 1
        ss.snap.residence_sum += self.now - frame.entry_time
        ss.snap.residence_cnt += 1
        ss._in_sys -= 1
        if frame.pool_wait_start is not None:
            ss.snap.pool_wait_time_sum += self.now - frame.pool_wait_start

        # 归还池令牌并唤醒一个等待者(它仍持有上层资源)
        if ss.tokens is not None:
            ss.tokens += 1
            if ss.pool_q:
                nxt = ss.pool_q.popleft()
                ss._pool_len -= 1
                wait_start = nxt.frames[-1].pool_wait_start
                if wait_start is not None:
                    ss.snap.pool_wait_time_sum += self.now - wait_start
                ss.tokens -= 1
                self._enter_service(nxt, station)

        if req.frames:
            # 从子调用返回父帧, 父帧继续发起下一个子调用或退出
            parent = req.frames[-1].station
            self._advance(req, parent)
            return
        # 纯嵌套模型: 入口帧退出后栈空 = 整条事务完成
        self._on_request_complete(req)

    def _on_request_complete(self, req: _Request) -> None:
        cname = req.cls.name
        rt = self.now - req.birth
        self._cls_resp[cname].append(rt)
        self._cls_count[cname] += 1
        if req.cls.kind == "closed":
            # 闭合类: 同一用户 think 后发起下一事务
            req.in_think = True
            self._push(self.now + self._think_time(req.cls.think_time),
                       _E_THINK_DONE, req)
        # 开放类的到达由独立的 _E_OPEN_ARRIVAL 链驱动, 此处不做任何调度

    # ------------------------------------------------------------------ #
    # warmup 标记与结果
    # ------------------------------------------------------------------ #
    def _mark_warmup(self) -> None:
        """在 warmup 结束时刻冻结累计基线, 之后的统计才计入结果。"""
        for ss in self.st.values():
            if ss.warm_snap is None:
                ss.mark_warm(self._warmup)
        self._cls_warm_count = dict(self._cls_count)
        self._cls_warm_sum = {
            c: float(np.sum(v)) for c, v in self._cls_resp.items()}
        self._root_warm = dict(self._root_arrivals)

    def result(self) -> "SimResult":
        if not self._end_time:
            raise RuntimeError("先 run()")
        T = self._horizon
        # 测量窗口内各类根到达率(无窗口截断偏置), 配合路由访问比给站点吞吐
        root_rate = {
            c: (self._root_arrivals[c] - self._root_warm[c]) / T
            for c in self.net.classes}
        visits = self.net.per_class_visits()
        station_metrics: Dict[str, Dict[str, float]] = {}
        for name, ss in self.st.items():
            d = ss.delta()
            m = ss.st.servers
            station_metrics[name] = {
                "utilization": (d.busy_area / (m * T)) if m is not None
                else d.busy_area / T,
                "mean_queue_length": d.queue_area / T,
                "mean_pool_waiters": d.pool_wait_area / T,
                "mean_in_station": d.in_system_area / T,
                "throughput": sum(root_rate[c] * visits[c][name]
                                  for c in self.net.classes),
                "mean_residence": (d.residence_sum / d.residence_cnt
                                   if d.residence_cnt else float("nan")),
                "mean_pool_wait": (d.pool_wait_time_sum / d.arrivals
                                   if d.arrivals else 0.0),
                "arrivals": float(d.arrivals),
            }
        class_metrics: Dict[str, Dict[str, float]] = {}
        for cname, cls in self.net.classes.items():
            resps = np.array(self._cls_resp[cname], dtype=float)
            n_done = self._cls_count[cname] - self._cls_warm_count[cname]
            warm_sum = self._cls_warm_sum[cname]
            total_sum = float(resps.sum())
            mean_rt = ((total_sum - warm_sum) / n_done
                       if n_done > 0 else float("nan"))
            n_root = (self._root_arrivals[cname]
                      - self._root_warm[cname])
            class_metrics[cname] = {
                "throughput": n_root / T,
                "response_time": mean_rt,
                "completions": float(n_done),
            }
        return SimResult(
            network_name=self.net.name,
            horizon=T,
            warmup=self._warmup,
            classes=class_metrics,
            stations=station_metrics,
        )


@dataclass
class SimResult:
    network_name: str
    horizon: float
    warmup: float
    classes: Dict[str, Dict[str, float]]
    stations: Dict[str, Dict[str, float]]
    replications: Optional[List["SimResult"]] = None
    ci: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None


# 学生 t 分位数(双侧 95%)
_T_TABLE = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
            7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
            13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101,
            19: 2.093, 20: 2.086}


def run_replications(net: Network, n_rep: int = 5, warmup: float = 2000,
                     horizon: float = 8000, base_seed: int = 1000,
                     rate_overrides: Optional[Dict[str, Callable[[float], float]]] = None,
                     ) -> SimResult:
    """多次独立重复仿真, 返回均值与 95% 置信半宽。"""
    reps: List[SimResult] = []
    for r in range(n_rep):
        sim = Simulator(net, seed=base_seed + r * 7919,
                        rate_overrides=rate_overrides)
        sim.run(warmup, horizon)
        reps.append(sim.result())

    def aggregate(group: str) -> Tuple[Dict[str, Dict[str, float]],
                                       Dict[str, Dict[str, float]]]:
        names = list(reps[0].__dict__[group].keys())
        means: Dict[str, Dict[str, float]] = {}
        cis: Dict[str, Dict[str, float]] = {}
        for name in names:
            means[name] = {}
            cis[name] = {}
            metrics = list(reps[0].__dict__[group][name].keys())
            for metric in metrics:
                sample = np.array([
                    getattr(r, group)[name][metric] for r in reps
                ], dtype=float)
                finite = sample[np.isfinite(sample)]
                if finite.size == 0:
                    means[name][metric] = float("nan")
                    cis[name][metric] = float("nan")
                    continue
                means[name][metric] = float(finite.mean())
                if n_rep > 1 and finite.size > 1:
                    sd = float(finite.std(ddof=1))
                    t = _T_TABLE.get(finite.size - 1, 1.96)
                    cis[name][metric] = t * sd / math.sqrt(finite.size)
                else:
                    cis[name][metric] = 0.0
        return means, cis

    cls_mean, cls_ci = aggregate("classes")
    st_mean, st_ci = aggregate("stations")
    return SimResult(
        network_name=net.name,
        horizon=horizon,
        warmup=warmup,
        classes=cls_mean,
        stations=st_mean,
        replications=reps,
        ci={"classes": cls_ci, "stations": st_ci},
    )
