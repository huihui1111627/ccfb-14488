# -*- coding: utf-8 -*-
"""不动产登记系统的示例排队网络场景。

拓扑（与省厅实际部署同构，数值为脱敏后的教学级参数）
====================================================

    用户/外部系统
        │
   [网关 gateway]  FCFS ×4，鉴权、限流、报文处理
        │
   [应用 app]      PS ×8（CPU 核数），受 app_threads 线程池(24)约束
        │
        ├──> [缓存 cache] FCFS ×2，Redis 集群
        ├──> [数据库 db]  FCFS ×8，受 db_connections 连接池(10)约束
        └──> [外部接口 ext] IS 纯延迟，税务/公安等外部系统回单延迟

四类业务
========
- register  登记发证（闭合，60 个内网用户）：写库为主，两次访问 DB，
  持有应用线程期间申请 DB 连接，连接池满时线程阻塞——嵌套占用的根源。
- query     查询核验（闭合，120 个用户）：读多写少，先查缓存再回源 DB。
- tax       税务联办（闭合，30 个联办窗口用户）：调用外部接口最多，
  纯延迟站点占主导。
- push      第三方推送（开放，泊松到达 18 个/秒）：外部系统回调写入，
  交税截止日前到达率会翻倍且呈强突发。

三档验收负载
============
- light  轻载：×0.5，最忙站点利用率 30%~45%
- medium 中载：×1.0，最忙站点利用率 60%~75%
- heavy  近饱和：×1.8，最忙站点/连接池利用率 ≥85%
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .network import (Station, Pool, Step, RequestClass, QueueNetwork,
                      FCFS, PS, IS)

STATION_SPECS = [
    # 名称, 服务台数, 规则, 说明
    ("gateway", 4, FCFS, "API 网关：鉴权限流"),
    ("app", 8, PS, "应用服务器 CPU（处理器共享）"),
    ("cache", 2, FCFS, "缓存集群"),
    ("db", 8, FCFS, "数据库服务器"),
    ("ext", 1, IS, "外部接口（纯延迟）"),
]

POOL_SPECS = [
    ("app_threads", 24, "应用线程池"),
    ("db_connections", 10, "数据库连接池"),
]


def build_network(population_scale: float = 1.0,
                  arrival_scale: float = 1.0,
                  servers_override: Dict[str, int] = None,
                  pool_override: Dict[str, int] = None) -> QueueNetwork:
    """构造不动产登记系统排队网络。"""
    stations = [
        Station(name,
                (servers_override or {}).get(name, m) if d != IS else 1,
                d, desc)
        for name, m, d, desc in STATION_SPECS]
    pools = [Pool(name,
                  (pool_override or {}).get(name, cap), desc)
             for name, cap, desc in POOL_SPECS]

    # -------------------------------------------------------------- #
    # 登记发证：网关→(申请线程)应用→缓存→(申请连接)数据库→外部→
    #           数据库(释放连接)→应用(释放线程)→网关
    # 注意：外部接口调用期间释放数据库连接（避免拿着连接干等外部系统），
    # ext 返回后第二次访问 DB 时重新申请——演示嵌套资源的分段持有。
    # -------------------------------------------------------------- #
    register = RequestClass(
        "register", "closed", think=18.0,
        population=max(1, round(60 * population_scale)),
        desc="登记发证（写业务）",
        route=[
            Step("gateway", 0.0260),
            Step("app", 0.0455, acquire="app_threads"),
            Step("cache", 0.0130),
            Step("db", 0.0715, acquire="db_connections",
                 release="db_connections"),
            Step("ext", 0.0780),
            Step("db", 0.1040, acquire="db_connections",
                 release="db_connections"),
            Step("app", 0.0325, release="app_threads"),
            Step("gateway", 0.0163),
        ])

    # 查询核验：网关→(线程)应用→缓存→缓存未命中回源 DB(连接)→应用(释放)→网关
    query = RequestClass(
        "query", "closed", think=10.0,
        population=max(1, round(120 * population_scale)),
        desc="查询核验（读业务）",
        route=[
            Step("gateway", 0.0195),
            Step("app", 0.0325, acquire="app_threads"),
            Step("cache", 0.0117),
            Step("db", 0.0520, acquire="db_connections",
                 release="db_connections"),
            Step("app", 0.0228, release="app_threads"),
            Step("gateway", 0.0130),
        ])

    # 税务联办：外部交互最多
    tax = RequestClass(
        "tax", "closed", think=22.0,
        population=max(1, round(30 * population_scale)),
        desc="税务联办（外部接口密集）",
        route=[
            Step("gateway", 0.0228),
            Step("app", 0.0390, acquire="app_threads"),
            Step("ext", 0.1300),
            Step("db", 0.0585, acquire="db_connections",
                 release="db_connections"),
            Step("ext", 0.0975),
            Step("app", 0.0260, release="app_threads"),
            Step("gateway", 0.0143),
        ])

    # 第三方推送（开放类）：截止日前突发主体
    push = RequestClass(
        "push", "open", arrival=26.0 * arrival_scale,
        desc="第三方系统推送（开放到达）",
        route=[
            Step("gateway", 0.0182),
            Step("app", 0.0390, acquire="app_threads"),
            Step("db", 0.0650, acquire="db_connections",
                 release="db_connections"),
            Step("app", 0.0195, release="app_threads"),
            Step("gateway", 0.0104),
        ])

    return QueueNetwork(stations, [register, query, tax, push],
                        pools=pools, name="不动产登记系统")


# ---------------------------------------------------------------------- #
# 三档负载
# ---------------------------------------------------------------------- #
LOAD_LEVELS: Dict[str, Tuple[float, float, str]] = {
    # 名称: (闭合用户倍数, 开放到达倍数, 说明)
    "light": (0.5, 0.5, "轻载，最忙资源利用率 20%~35%"),
    "medium": (1.0, 1.0, "中载，最忙资源利用率 40%~55%"),
    "heavy": (1.8, 2.1, "近饱和，数据库利用率 ≈73%，连接池 ≈58%"),
}


def load_network(level: str) -> QueueNetwork:
    if level not in LOAD_LEVELS:
        raise KeyError(f"未知负载档位: {level}，可选 {list(LOAD_LEVELS)}")
    p_scale, a_scale, _ = LOAD_LEVELS[level]
    return build_network(population_scale=p_scale, arrival_scale=a_scale)


# ---------------------------------------------------------------------- #
# 合成监控数据（用于需求反推的演示与测试）
# ---------------------------------------------------------------------- #
def synthetic_monitor(net: QueueNetwork, n_windows: int = 24,
                      noise: float = 0.03, seed: int = 7
                      ) -> List["MonitorSample"]:
    """生成带噪声的合成监控窗口数据。

    真实环境里监控只给得出利用率和吞吐，服务需求要靠 :mod:`estimation`
    反推。为让超定方程良态（避免各类吞吐完全共线），每个窗口对每一类
    施加独立的负载因子——对应真实场景里登记、查询、联办、推送各自的
    业务高峰并不同步。利用率按真实需求与该窗口吞吐正演，再加观测噪声，
    模拟 APM 导出。
    """
    from .estimation import MonitorSample
    from .layered import LayeredSolver

    rng = np.random.default_rng(seed)
    D = net.demands
    # 参考稳态：取基准网络的解作为各窗口吞吐的中枢
    base = LayeredSolver(net).solve()
    samples = []
    for w in range(n_windows):
        # 每类独立缩放，保证吞吐矩阵列满秩、需求可辨识
        scales = rng.uniform(0.45, 1.25, size=net.C)
        thr = {}
        for ci, c in enumerate(net.classes):
            x = base.X[ci] * scales[ci] * float(rng.lognormal(0, noise))
            thr[c.name] = max(1e-6, x)
        # 利用率正演：排队站 U[s]=Σ_c X[c]D[c,s]（D 为每轮总服务时间，
        # 已隐含除以服务台数的占比口径）；IS 站报告 erlang 负荷
        xvec = np.array([thr[c.name] for c in net.classes])
        util = {}
        for si, st in enumerate(net.stations):
            u = float(np.sum(xvec * D[:, si]))
            if st.discipline != IS:
                u = u / st.servers
            util[st.name] = min(0.999, max(0.0,
                                u * float(rng.lognormal(0, noise))))
        resp = {c.name: float(base.R[ci])
                for ci, c in enumerate(net.classes)}
        pop = {c.name: c.population
               for c in net.classes if c.kind == "closed"}
        samples.append(MonitorSample(thr, util, resp, pop))
    return samples
