"""分层排队模型: 线程池/连接池嵌套占用的处理。

真实系统里应用线程在等待数据库连接/缓存返回期间**不释放**。
本模块采用稳健的**基线 MVA + 池阻塞增量**建模, 不做容易在饱和区
正反馈发散的延迟不动点:

1. 在正确展开的需求矩阵 D 上构造平铺网络, 服务台数/调度保持原值,
   先跑标准近似 MVA(相当于假设无限资源池);
2. 对每个带池站点, 用工作点并发作为提供负载, Erlang-B 估计令牌全
   占满的阻塞概率、Erlang-C 估计必须等待的概率;
3. 池外排队人口 = P(wait) * 超出池容量的在站人口, 除以池的有效吞吐
   cap/s 得到每次访问的池等待延迟, 按各类访问次数加到端到端响应;
4. 池容量与服务台数相互独立(池可以大于服务器数), 嵌套持有期间父站
   服务台已释放、令牌不释放, 平铺 D 已正确区分这两部分需求。

无嵌套边或无池时, 结果与 :func:`mva.solve_approx` 一致。
深度硬饱和区绝对值偏保守, 精确预测以仿真为准(见 README 已知边界)。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


from .network import (Network, Station, ServiceClass, RouteStep, CallSpec,
                      SCHED_IS)
from .mva import ModelResult, solve_approx, erlang_c


@dataclass
class PoolDiagnostic:
    station: str
    servers: int
    pool_capacity: int
    effective_servers: int
    blocking_prob: float
    mean_concurrency: float


def _nested_edges(net: Network) -> List[Tuple[str, str]]:
    edges, seen = [], set()
    for cls in net.classes.values():
        for step in cls.route:
            for call in step.calls:
                key = (step.station, call.callee)
                if key not in seen:
                    seen.add(key)
                    edges.append(key)
    return edges


def erlang_b(m: int, a: float) -> float:
    """Erlang-B 阻塞概率: m 条并发线、提供负载 a 时全部占满的概率。"""
    if a <= 0:
        return 0.0
    inv = 1.0
    for k in range(1, m + 1):
        inv = 1.0 + inv * k / a
    return 1.0 / inv


def _flatten(net: Network) -> Network:
    """用展开需求 D 构造等价平铺网络(服务台数/调度保持原值)。

    顺序路由步用 CallSpec(1) 串成调用链, 使纯嵌套展开覆盖每站。
    池约束不在这里改台数, 而在求解后用 Erlang-B 阻塞增量修正。
    """
    D = net.per_class_demands()
    stations: List[Station] = []
    for s in net.stations.values():
        if s.sched == SCHED_IS:
            stations.append(Station(s.name, s.sched, servers=None,
                                    pool_capacity=None))
        else:
            stations.append(Station(s.name, s.sched,
                                    servers=s.servers, pool_capacity=None))
    classes: List[ServiceClass] = []
    for c in net.classes.values():
        visited = [s for s in net.stations if D[c.name][s] > 1e-12]
        steps: List[RouteStep] = []
        for idx, s in enumerate(visited):
            calls = ((CallSpec(visited[idx + 1], 1),)
                     if idx + 1 < len(visited) else ())
            steps.append(RouteStep(s, D[c.name][s], calls=calls))
        classes.append(ServiceClass(
            name=c.name, kind=c.kind, route=steps,
            population=c.population, think_time=c.think_time,
            arrival_rate=c.arrival_rate, cv=c.cv, color=c.color))
    return Network(net.name, stations, classes)


def solve_layered(net: Network, max_iter: int = 100,
                  tol: float = 1e-5,
                  open_rates: Optional[Dict[str, float]] = None
                  ) -> ModelResult:
    """分层求解带嵌套资源池的网络。

    先在展开需求上跑标准近似 MVA(忽略池), 再对每个带池站点用
    Erlang-B/C 估计池阻塞, 把"持上层资源等令牌"的延迟增量按各类
    访问次数加到端到端响应上。池容量与服务台数相互独立。
    """
    edges = _nested_edges(net)
    has_pool = any(s.pool_capacity is not None
                   for s in net.stations.values())

    work = _flatten(net)
    result = solve_approx(work, max_iter=max_iter, tol=tol,
                          open_rates=open_rates)

    if not edges or not has_pool:
        return result

    result.method = "layered FES MVA"
    D = net.per_class_demands()
    V = net.per_class_visits()
    # 每个带池站点的池等待延迟(每次访问该站的平均附加秒数)
    pool_wait: Dict[str, float] = {}
    diag: List[PoolDiagnostic] = []
    notes: List[str] = []
    for name, s in net.stations.items():
        if s.pool_capacity is None or s.sched == SCHED_IS:
            continue
        cap = s.pool_capacity
        servers = s.servers or 1
        util = result.station_utilization.get(name, 0.0)
        in_service = util * servers
        queue = result.station_queue_length.get(name, 0.0)
        base_concurrency = max(in_service + queue, in_service)
        a = base_concurrency
        p_block = erlang_b(cap, a)
        p_wait = erlang_c(cap, min(a, cap - 1e-9))
        # 池外等待者: 仅当基线并发逼近池容量时才显著。
        # 用 Erlang-C 等待概率对"超出 cap 的在站人口"加权,
        # 不引入闭合总人口这个过粗的上界, 避免轻微阻塞被放大。
        excess = max(base_concurrency - cap, 0.0)
        outside_now = p_wait * excess
        concurrency = base_concurrency + outside_now
        # 阻塞者人均等待 = 池外排队人口 / 池的有效吞吐(cap/s)
        s_station = _weighted_service(net, result, name)
        x_cap = cap / max(s_station, 1e-9)
        wait = outside_now / max(x_cap, 1e-9)
        pool_wait[name] = wait
        diag.append(PoolDiagnostic(
            station=name, servers=servers, pool_capacity=cap,
            effective_servers=min(cap, servers),
            blocking_prob=p_block, mean_concurrency=concurrency))
        notes.append(
            f"{name}:pool={cap},P(block)={p_block:.3f},"
            f"P(wait)={p_wait:.3f},wait/visit={wait:.4f}s")

    # 把池等待按各类在该站的访问次数加到响应时间与驻留
    for cname in net.classes:
        extra = sum(V[cname].get(name, 0.0) * w
                    for name, w in pool_wait.items())
        result.class_response[cname] += extra
        for name, w in pool_wait.items():
            if name in result.per_class_residence[cname]:
                result.per_class_residence[cname][name] += (
                    V[cname].get(name, 0.0) * w)
    result.warnings.append("pool=" + ";".join(notes))
    result.pool_diagnostics = diag  # type: ignore[attr-defined]
    return result


def _weighted_service(net: Network, result: ModelResult,
                      station: str) -> float:
    """站点按类吞吐加权的平均单次服务时间。"""
    D = net.per_class_demands()
    V = net.per_class_visits()
    num = den = 0.0
    for cname in net.classes:
        x = result.class_throughput[cname]
        v = V[cname].get(station, 0.0)
        if x > 0 and v > 0:
            num += x * v * (D[cname][station] / v)
            den += x * v
    return num / den if den > 0 else 0.0
