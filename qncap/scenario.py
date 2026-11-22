"""不动产登记系统参考场景。

五站点拓扑(与生产一致)::

    用户 -> 网关(FCFS) -> 应用(PS + 线程池)
                            ├─ 缓存(IS 纯延迟, 命中部分)
                            ├─ 数据库(FCFS + 连接池)
                            └─ 外部接口(FCFS, 税务/住建共享接口)

四类负载:

* 查询类(closed): 登记信息查询, 读多写少, 大量走缓存;
* 登簿类(closed): 转移/抵押登记, 写库 + 多次外部核验, 最重;
* 网上预约(open): 外部互联网渠道持续进入;
* 批量上报(open): 地市系统定时批量推送, 突发性最强。

提供两个网络:
* :func:`build_flat_network` —— 平铺网络(无池限制), 用于精确 MVA 验收;
* :func:`build_layered_network` —— 带线程池/连接池的分层网络,
  复刻"应用扩容一倍却卡在数据库连接池"的生产事故形态。
"""

from __future__ import annotations

from typing import Dict, Optional

from .network import (Network, Station, ServiceClass, RouteStep, CallSpec,
                      SCHED_FCFS, SCHED_PS, SCHED_IS)

# 站点默认服务台数
DEFAULT_SERVERS = {
    "gateway": 4,
    "app": 8,
    "cache": None,       # IS
    "db": 6,
    "ext": 3,
}

# 线程池/连接池容量(分层场景)
DEFAULT_POOLS = {
    "app": 40,
    "db": 30,
}


def _stations(pools: bool) -> list:
    cap = DEFAULT_POOLS if pools else {}
    return [
        Station("gateway", SCHED_FCFS, servers=DEFAULT_SERVERS["gateway"]),
        Station("app", SCHED_PS, servers=DEFAULT_SERVERS["app"],
                pool_capacity=cap.get("app")),
        Station("cache", SCHED_IS),
        Station("db", SCHED_FCFS, servers=DEFAULT_SERVERS["db"],
                pool_capacity=cap.get("db")),
        Station("ext", SCHED_FCFS, servers=DEFAULT_SERVERS["ext"]),
    ]


def _classes(closed_pop: Optional[Dict[str, int]] = None,
             open_rates: Optional[Dict[str, float]] = None,
             cv: float = 1.0) -> list:
    pop = closed_pop or {"query": 60, "register": 24}
    rates = open_rates or {"booking": 6.0, "bulk": 1.5}

    # 建模约定: 路由第一步是入口(网关), 网关同步调用应用,
    # 应用再同步调用缓存/库/外部接口; 其余路由步是被调站点的行为定义。
    # ---- 闭合类 1: 查询 ----
    query = ServiceClass(
        name="query", kind="closed",
        population=pop["query"], think_time=5.0, cv=cv, color="#4C78A8",
        route=[
            RouteStep("gateway", 0.012, calls=(CallSpec("app", 1),)),
            RouteStep("app", 0.030, calls=(
                CallSpec("cache", 2),
                CallSpec("db", 0.6),
                CallSpec("ext", 0.2),
            )),
            RouteStep("cache", 0.006),
            RouteStep("db", 0.045),
            RouteStep("ext", 0.090),
        ])

    # ---- 闭合类 2: 登簿(转移/抵押) ----
    register = ServiceClass(
        name="register", kind="closed",
        population=pop["register"], think_time=8.0, cv=cv, color="#F58518",
        route=[
            RouteStep("gateway", 0.015, calls=(CallSpec("app", 1),)),
            RouteStep("app", 0.060, calls=(
                CallSpec("cache", 1),
                CallSpec("db", 3),
                CallSpec("ext", 2),
            )),
            RouteStep("cache", 0.008),
            RouteStep("db", 0.080),
            RouteStep("ext", 0.120),
        ])

    # ---- 开放类 1: 网上预约 ----
    booking = ServiceClass(
        name="booking", kind="open",
        arrival_rate=rates["booking"], cv=cv, color="#54A24B",
        route=[
            RouteStep("gateway", 0.010, calls=(CallSpec("app", 1),)),
            RouteStep("app", 0.025, calls=(
                CallSpec("cache", 1),
                CallSpec("db", 1),
            )),
            RouteStep("cache", 0.005),
            RouteStep("db", 0.050),
        ])

    # ---- 开放类 2: 批量上报(重库写) ----
    bulk = ServiceClass(
        name="bulk", kind="open",
        arrival_rate=rates["bulk"], cv=1.6, color="#E45756",
        route=[
            RouteStep("gateway", 0.020, calls=(CallSpec("app", 1),)),
            RouteStep("app", 0.100, calls=(
                CallSpec("db", 4),
                CallSpec("ext", 1),
            )),
            RouteStep("db", 0.110),
            RouteStep("ext", 0.150),
        ])
    return [query, register, booking, bulk]


def build_flat_network(closed_pop: Optional[Dict[str, int]] = None,
                       open_rates: Optional[Dict[str, float]] = None,
                       cv: float = 1.0) -> Network:
    """无资源池限制的平铺网络(精确 MVA 可直接解)。"""
    return Network("不动产登记系统(平铺模型)",
                   _stations(pools=False),
                   _classes(closed_pop, open_rates, cv=cv))


def build_layered_network(closed_pop: Optional[Dict[str, int]] = None,
                          open_rates: Optional[Dict[str, float]] = None,
                          cv: float = 1.0,
                          app_pool: int = 40,
                          db_pool: int = 30) -> Network:
    """带线程池/连接池的分层网络(嵌套占用)。"""
    net = Network("不动产登记系统(分层池化模型)",
                  _stations(pools=True),
                  _classes(closed_pop, open_rates, cv=cv))
    net.stations["app"].pool_capacity = app_pool
    net.stations["db"].pool_capacity = db_pool
    return net


# 三档负载(轻载/中载/接近饱和), 闭合人口与开放到达率同步放大
LOAD_LEVELS = {
    "light":  {"query": 25, "register": 10},
    "medium": {"query": 60, "register": 24},
    "heavy":  {"query": 120, "register": 48},
}
OPEN_LEVELS = {
    "light":  {"booking": 3.0, "bulk": 0.75},
    "medium": {"booking": 6.0, "bulk": 1.5},
    "heavy":  {"booking": 11.0, "bulk": 3.0},
}


def build_scenario(level: str = "medium", layered: bool = False,
                   cv: float = 1.0) -> Network:
    if level not in LOAD_LEVELS:
        raise ValueError(f"负载档位必须是 {list(LOAD_LEVELS)}")
    builder = build_layered_network if layered else build_flat_network
    return builder(LOAD_LEVELS[level], OPEN_LEVELS[level], cv=cv)
