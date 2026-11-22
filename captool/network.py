# -*- coding: utf-8 -*-
"""排队网络模型定义。

三类排队规则
------------
- FCFS: 先到先服务，多服务台（``servers=m``）
- PS:   处理器共享，多服务台，容量被在站请求均分
- IS:   纯延迟（无限服务台，每个请求独立等待一段延迟，不排队）

请求类
------
- closed: 固定数量 ``population`` 个用户，走完一条路由后经历均值 ``think``
          的思考时间再发起下一轮（终端用户型业务）
- open:   外部按到达率 ``arrival``（个/秒）不断进入的业务

资源池（连接池/线程池）
-----------------------
路由步 :class:`Step` 可在进入站点前 ``acquire`` 一个资源池令牌、在步结束时
``release``。令牌未释放期间进入后续站点即构成“拿着上一层资源等下一层”的
嵌套占用；池满时请求在池的 FIFO 队列中等待，且继续持有已获得的上层资源。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

FCFS = "FCFS"
PS = "PS"
IS = "IS"
DISCIPLINES = (FCFS, PS, IS)
CLOSED = "closed"
OPEN = "open"


@dataclass
class Station:
    """排队站点。"""
    name: str
    servers: int          # 服务台数；IS 站点忽略（视为无限）
    discipline: str       # FCFS / PS / IS
    desc: str = ""

    def __post_init__(self):
        if self.discipline not in DISCIPLINES:
            raise ValueError(f"未知排队规则: {self.discipline}")
        if self.discipline != IS and self.servers <= 0:
            raise ValueError(f"站点 {self.name} 服务台数必须为正")


@dataclass
class Pool:
    """容量有限的资源池（数据库连接池、应用线程池等）。"""
    name: str
    capacity: int
    desc: str = ""


@dataclass
class Step:
    """路由中的一步：在某站点接受一次服务，可伴随资源池令牌的申请/释放。"""
    station: str
    service: float                       # 平均服务时间（秒，指数分布）
    acquire: Optional[str] = None        # 进入该步前申请的资源池
    release: Optional[str] = None        # 该步服务结束后释放的资源池


@dataclass
class RequestClass:
    """一类请求。"""
    name: str
    kind: str                             # closed / open
    route: List[Step] = field(default_factory=list)
    population: int = 0                   # closed: 固定用户数
    think: float = 0.0                    # closed: 平均思考时间（秒）
    arrival: float = 0.0                  # open: 平均到达率（个/秒）
    desc: str = ""

    def __post_init__(self):
        if self.kind not in (CLOSED, OPEN):
            raise ValueError(f"未知请求类类型: {self.kind}")
        if self.kind == CLOSED and self.population <= 0:
            raise ValueError(f"闭合类 {self.name} 需要正的 population")
        if self.kind == OPEN and self.arrival <= 0:
            raise ValueError(f"开放类 {self.name} 需要正的 arrival")


class QueueNetwork:
    """一张完整的排队网络。

    解析求解时以“每轮次/每到达的总服务需求” ``D[c,s]`` 和访问次数
    ``V[c,s]`` 为基本量；嵌套占用由 :mod:`captool.layered` 展开为资源池站点。
    """

    def __init__(self,
                 stations: Sequence[Station],
                 classes: Sequence[RequestClass],
                 pools: Optional[Sequence[Pool]] = None,
                 name: str = "network"):
        self.name = name
        self.stations: List[Station] = list(stations)
        self.classes: List[RequestClass] = list(classes)
        self.pools: List[Pool] = list(pools or [])
        self._validate()
        self._V, self._D = self._build_matrices()

    # ------------------------------------------------------------------ #
    # 基础校验与索引
    # ------------------------------------------------------------------ #
    def _validate(self):
        names = [s.name for s in self.stations]
        if len(set(names)) != len(names):
            raise ValueError("站点名重复")
        pool_names = {p.name for p in self.pools}
        cls_names = [c.name for c in self.classes]
        if len(set(cls_names)) != len(cls_names):
            raise ValueError("请求类名重复")
        station_set = set(names)
        for c in self.classes:
            if not c.route:
                raise ValueError(f"类 {c.name} 路由为空")
            for st in c.route:
                if st.station not in station_set:
                    raise ValueError(f"类 {c.name} 引用了不存在的站点 {st.station}")
                if st.acquire and st.acquire not in pool_names:
                    raise ValueError(f"类 {c.name} 引用了不存在的资源池 {st.acquire}")
                if st.release and st.release not in pool_names:
                    raise ValueError(f"类 {c.name} 引用了不存在的资源池 {st.release}")
        # 资源池的 acquire/release 必须在每条路由里成对
        for c in self.classes:
            held: List[str] = []
            for st in c.route:
                if st.acquire:
                    if st.acquire in held:
                        raise ValueError(f"类 {c.name} 重复申请池 {st.acquire}")
                    held.append(st.acquire)
                if st.release:
                    if st.release not in held:
                        raise ValueError(
                            f"类 {c.name} 释放了未持有的池 {st.release}")
                    held.remove(st.release)
            if held:
                raise ValueError(f"类 {c.name} 路由结束时仍持有池: {held}")

    @property
    def S(self) -> int:
        return len(self.stations)

    @property
    def C(self) -> int:
        return len(self.classes)

    def si(self, name: str) -> int:
        return self._station_idx[name]

    def ci(self, name: str) -> int:
        return self._class_idx[name]

    def _build_matrices(self):
        self._station_idx = {s.name: i for i, s in enumerate(self.stations)}
        self._class_idx = {c.name: i for i, c in enumerate(self.classes)}
        V = np.zeros((self.C, self.S))
        D = np.zeros((self.C, self.S))
        for ci, c in enumerate(self.classes):
            for step in c.route:
                si = self._station_idx[step.station]
                V[ci, si] += 1.0
                D[ci, si] += step.service
        return V, D

    # ------------------------------------------------------------------ #
    # 矩阵视图
    # ------------------------------------------------------------------ #
    @property
    def visits(self) -> np.ndarray:
        """访问次数矩阵 V[c,s]（每个闭合用户一轮 / 每个开放请求一次到达）。"""
        return self._V.copy()

    @property
    def demands(self) -> np.ndarray:
        """总服务需求矩阵 D[c,s] = Σ 访问服务时间（秒）。"""
        return self._D.copy()

    @property
    def closed_idx(self) -> List[int]:
        return [i for i, c in enumerate(self.classes) if c.kind == CLOSED]

    @property
    def open_idx(self) -> List[int]:
        return [i for i, c in enumerate(self.classes) if c.kind == OPEN]

    def populations(self) -> np.ndarray:
        return np.array([c.population if c.kind == CLOSED else 0
                         for c in classes_shim(self)], dtype=float)

    def thinks(self) -> np.ndarray:
        return np.array([c.think if c.kind == CLOSED else 0.0
                         for c in self.classes], dtype=float)

    def arrivals(self) -> np.ndarray:
        return np.array([c.arrival if c.kind == OPEN else 0.0
                         for c in self.classes], dtype=float)

    # ------------------------------------------------------------------ #
    # 资源池嵌套结构（供 layered 求解器使用）
    # ------------------------------------------------------------------ #
    def pool_hold_steps(self) -> Dict[str, List[Tuple[int, int]]]:
        """返回每个池在持有期间覆盖的 (类序号, 路由步序号) 列表。

        池令牌在某步 acquire，直到对应 release 步结束，期间经过的所有
        站点步都属于嵌套占用区间（release 步本身也在区间内）。
        """
        out: Dict[str, List[Tuple[int, int]]] = {p.name: [] for p in self.pools}
        for ci, c in enumerate(self.classes):
            held: List[str] = []
            for ki, step in enumerate(c.route):
                if step.acquire:
                    held.append(step.acquire)
                for pname in held:
                    out[pname].append((ci, ki))
                if step.release:
                    held.remove(step.release)
        return out

    def pool_acquire_visits(self) -> Dict[str, np.ndarray]:
        """每个池被各类申请的次数（每轮/每到达）。"""
        out = {p.name: np.zeros(self.C) for p in self.pools}
        for ci, c in enumerate(self.classes):
            for step in c.route:
                if step.acquire:
                    out[step.acquire][ci] += 1.0
        return out

    # ------------------------------------------------------------------ #
    # 场景复制（容量问答用）
    # ------------------------------------------------------------------ #
    def copy(self,
             population_scale: float = 1.0,
             arrival_scale: float = 1.0,
             servers_override: Optional[Dict[str, int]] = None,
             pool_override: Optional[Dict[str, int]] = None) -> "QueueNetwork":
        """复制网络并缩放用户数/到达率，或覆盖站点服务台数与池容量。"""
        stations = [Station(s.name,
                            servers_override.get(s.name, s.servers)
                            if servers_override and s.discipline != IS
                            else s.servers,
                            s.discipline, s.desc)
                    for s in self.stations]
        pools = [Pool(p.name,
                      pool_override.get(p.name, p.capacity)
                      if pool_override else p.capacity,
                      p.desc) for p in self.pools]
        classes = []
        for c in self.classes:
            nc = RequestClass(
                c.name, c.kind,
                [Step(st.station, st.service, st.acquire, st.release)
                 for st in c.route],
                population=max(1, round(c.population * population_scale))
                if c.kind == CLOSED else 0,
                think=c.think,
                arrival=c.arrival * arrival_scale if c.kind == OPEN else 0.0,
                desc=c.desc)
            classes.append(nc)
        return QueueNetwork(stations, classes, pools, name=self.name)


def classes_shim(net: QueueNetwork):
    return net.classes
