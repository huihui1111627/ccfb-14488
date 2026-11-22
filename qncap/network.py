"""排队网络数据模型。

网络由两类元素构成:

* **站点(Station)**: 网关 / 应用 / 缓存 / 数据库 / 外部接口,
  各自有调度规则 (FCFS 先到先服务、PS 处理器共享、IS 纯延迟)
  和服务台数量; 站点可带一个有限容量的资源池 (线程池/连接池)。
* **请求类(ServiceClass)**: 每个类有自己的路由 (一组路由步),
  路由步上可以挂**同步嵌套调用**——请求在父站点占着服务台,
  去子站点执行, 子站点返回后父站点继续服务。这正是
  "拿着一层的资源去等下一层" 的连接池/线程池场景。

闭合类(closed): 固定数量用户反复发起, 两次请求之间有 think time;
开放类(open): 外部按到达率不断进入。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

SCHED_FCFS = "fcfs"
SCHED_PS = "ps"
SCHED_IS = "is"


@dataclass(frozen=True)
class CallSpec:
    """一次同步嵌套调用: 访问 ``callee`` 站点 ``calls`` 次。

    被调用站点上干什么由该请求类自己在该站点的路由步决定
    (自带服务需求与更下层调用), 因此同一调用对不同类可以有不同耗时。
    """

    callee: str
    calls: float = 1.0


@dataclass(frozen=True)
class RouteStep:
    """请求类在某个站点上的一次访问。

    ``demand`` 是该站点自身的服务需求(秒, 单服务台口径);
    ``calls`` 是服务期间发起的同步子调用。
    """

    station: str
    demand: float
    calls: Tuple[CallSpec, ...] = ()


@dataclass
class Station:
    """排队站点。

    ====== =================================================================
    字段   含义
    ====== =================================================================
    name   站点名
    sched  调度规则: ``fcfs`` / ``ps`` / ``is``
    servers 服务台数; IS(纯延迟)站点为 ``None``(无限台)
    pool_capacity 资源池容量(线程池/连接池); ``None`` 表示无额外限制,
                   即并发上限就是 servers; 设置后请求必须先拿到池令牌
                   才能接受服务, 拿不到时在父站点持有上层资源排队
    ====== =================================================================
    """

    name: str
    sched: str
    servers: Optional[int] = 1
    pool_capacity: Optional[int] = None

    def __post_init__(self) -> None:
        if self.sched not in (SCHED_FCFS, SCHED_PS, SCHED_IS):
            raise ValueError(f"未知调度规则: {self.sched}")
        if self.sched == SCHED_IS:
            self.servers = None
        elif self.servers is None or self.servers < 1:
            raise ValueError(f"站点 {self.name} 的服务台数必须 >=1")
        if self.pool_capacity is not None and self.pool_capacity < 1:
            raise ValueError(f"站点 {self.name} 的池容量必须 >=1")

    @property
    def is_infinite(self) -> bool:
        return self.sched == SCHED_IS


@dataclass
class ServiceClass:
    """一类请求。

    kind='closed' 时用 population / think_time;
    kind='open'   时用 arrival_rate(每秒请求数)。
    cv 为服务时间变异系数(1=指数分布, 0=定长, 其余按 Gamma 分布)。
    """

    name: str
    kind: str
    route: List[RouteStep]
    population: Optional[int] = None
    think_time: float = 0.0
    arrival_rate: Optional[float] = None
    cv: float = 1.0
    color: str = "#4C78A8"

    def __post_init__(self) -> None:
        if self.kind not in ("open", "closed"):
            raise ValueError(f"类 {self.name} 的 kind 必须是 open/closed")
        if self.kind == "closed":
            if not self.population or self.population < 1:
                raise ValueError(f"闭合类 {self.name} 需要 population>=1")
            if self.think_time < 0:
                raise ValueError(f"闭合类 {self.name} think_time 不能为负")
        else:
            if self.arrival_rate is None or self.arrival_rate <= 0:
                raise ValueError(f"开放类 {self.name} 需要正的 arrival_rate")
        if self.cv < 0:
            raise ValueError(f"类 {self.name} 的 cv 不能为负")

    def step(self, station: str) -> Optional[RouteStep]:
        for st in self.route:
            if st.station == station:
                return st
        return None


class Network:
    """一张排队网络: 站点 + 请求类 + 调用关系。"""

    def __init__(self, name: str, stations: List[Station],
                 classes: List[ServiceClass]) -> None:
        self.name = name
        self.stations: Dict[str, Station] = {}
        self.classes: Dict[str, ServiceClass] = {}
        for s in stations:
            if s.name in self.stations:
                raise ValueError(f"站点重名: {s.name}")
            self.stations[s.name] = s
        for c in classes:
            if c.name in self.classes:
                raise ValueError(f"请求类重名: {c.name}")
            self.classes[c.name] = c
        self._validate_refs()
        self._check_acyclic_calls()

    # ------------------------------------------------------------------ #
    # 基础校验
    # ------------------------------------------------------------------ #
    def _validate_refs(self) -> None:
        for c in self.classes.values():
            for st in c.route:
                if st.station not in self.stations:
                    raise ValueError(
                        f"类 {c.name} 路由引用了不存在的站点 {st.station}")
                if st.demand < 0:
                    raise ValueError(f"类 {c.name} 在 {st.station} 需求为负")
                for call in st.calls:
                    if call.callee not in self.stations:
                        raise ValueError(
                            f"类 {c.name} 调用了不存在的站点 {call.callee}")
                    if call.calls < 0:
                        raise ValueError(f"类 {c.name} 调用次数为负")

    def _check_acyclic_calls(self) -> None:
        """嵌套调用图必须无环(分层模型前提; 环网请用路由概率表达)。"""
        color: Dict[str, int] = {s: 0 for s in self.stations}

        def dfs(u: str) -> None:
            color[u] = 1
            for c in self.classes.values():
                step = c.step(u)
                if step is None:
                    continue
                for call in step.calls:
                    if color[call.callee] == 1:
                        raise ValueError(
                            f"嵌套调用图存在环: {u} -> {call.callee}")
                    if color[call.callee] == 0:
                        dfs(call.callee)
            color[u] = 2

        for s in self.stations:
            if color[s] == 0:
                dfs(s)

    # ------------------------------------------------------------------ #
    # 调用图 / 分层
    # ------------------------------------------------------------------ #
    def call_edges(self) -> List[Tuple[str, str]]:
        """去重后的调用边 (caller -> callee), 用于画图与分层。

        包含两类边: 路由顺序边(类路由中相邻站点)与同步嵌套调用边。
        """
        edges = []
        seen = set()
        for c in self.classes.values():
            seq = [st.station for st in c.route]
            for a, b in zip(seq, seq[1:]):
                key = (a, b)
                if key not in seen:
                    seen.add(key)
                    edges.append(key)
            for st in c.route:
                for call in st.calls:
                    key = (st.station, call.callee)
                    if key not in seen:
                        seen.add(key)
                        edges.append(key)
        return edges

    def station_layers(self) -> Dict[str, int]:
        """按调用图最长路径给站点分层(根=0层)。没有入边的站点在第0层。"""
        indeg = {s: 0 for s in self.stations}
        adj: Dict[str, List[str]] = {s: [] for s in self.stations}
        for u, v in self.call_edges():
            adj[u].append(v)
            indeg[v] += 1
        layer = {s: 0 for s in self.stations}
        queue = [s for s, d in indeg.items() if d == 0]
        for u in queue:
            for v in adj[u]:
                layer[v] = max(layer[v], layer[u] + 1)
                indeg[v] -= 1
                if indeg[v] == 0:
                    queue.append(v)
        return layer

    def root_stations(self) -> List[str]:
        """路由中可作为请求入口的站点(类路由的第一步)。"""
        roots = []
        for c in self.classes.values():
            first = c.route[0].station
            if first not in roots:
                roots.append(first)
        return roots

    # ------------------------------------------------------------------ #
    # 需求与访问量(展开嵌套调用后的口径)
    #
    # 建模约定: 类路由的第一个路由步是事务入口; 其余路由步描述对应站点
    # "被访问时的行为"(自身需求 + 同步子调用)。站点之间的流转全部通过
    # CallSpec 嵌套调用表达(含网关->应用这种顺序调用), 因此从入口步
    # 纯递归即可得到完整访问量, 不会重复计数。
    # ------------------------------------------------------------------ #
    def _walk_weighted(self, station: str, cls: ServiceClass,
                       visits: Dict[str, float], demands: Dict[str, float],
                       stack: Tuple[str, ...], weight: float) -> None:
        if station in stack:
            raise ValueError("嵌套调用出现环")
        step = cls.step(station)
        if step is None:
            return
        visits[station] = visits.get(station, 0.0) + weight
        demands[station] = demands.get(station, 0.0) + weight * step.demand
        for call in step.calls:
            self._walk_weighted(call.callee, cls, visits, demands,
                                stack + (station,), weight * call.calls)

    def _expand(self, cls: ServiceClass
                ) -> Tuple[Dict[str, float], Dict[str, float]]:
        visits: Dict[str, float] = {}
        demands: Dict[str, float] = {}
        self._walk_weighted(cls.route[0].station, cls, visits, demands,
                            (), 1.0)
        return visits, demands

    def per_class_visits(self) -> Dict[str, Dict[str, float]]:
        """每个请求完成一次完整事务, 在各站点的期望访问次数 V[k,i]。"""
        out: Dict[str, Dict[str, float]] = {}
        for cname, cls in self.classes.items():
            visits, _ = self._expand(cls)
            out[cname] = {s: visits.get(s, 0.0) for s in self.stations}
        return out

    def per_class_demands(self) -> Dict[str, Dict[str, float]]:
        """每个事务在各站点的总服务需求 D[k,i](秒)。"""
        out: Dict[str, Dict[str, float]] = {}
        for cname, cls in self.classes.items():
            _, demands = self._expand(cls)
            out[cname] = {s: demands.get(s, 0.0) for s in self.stations}
        return out

    def own_demand_matrix(self) -> Dict[str, Dict[str, float]]:
        """不含子调用的站点自身需求矩阵(分层模型用)。"""
        return {
            cname: {
                s: (cls.step(s).demand if cls.step(s) is not None else 0.0)
                for s in self.stations
            }
            for cname, cls in self.classes.items()
        }

    def call_counts(self) -> Dict[Tuple[str, str], Dict[str, float]]:
        """每条调用边上, 各类每次父访问的调用次数。"""
        out: Dict[Tuple[str, str], Dict[str, float]] = {}
        for cname, cls in self.classes.items():
            for st in cls.route:
                for call in st.calls:
                    key = (st.station, call.callee)
                    out.setdefault(key, {})
                    out[key][cname] = call.calls
        return out

    def copy(self) -> "Network":
        import copy
        return copy.deepcopy(self)
