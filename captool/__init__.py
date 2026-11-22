# -*- coding: utf-8 -*-
"""captool —— 基于排队网络的容量评估工具包。

模块划分
--------
- network:    排队网络（站点/资源池/请求类/路由）的数据定义
- mva:        解析求解（Erlang-C、混合开闭网络的精确 MVA 与 Schweitzer 近似 MVA）
- layered:    分层排队模型，处理连接池/线程池的嵌套资源占用
- estimation: 由监控利用率与吞吐反推各类请求在各站点的服务需求
- burst:      突发到达与非平稳到达的修正
- sizing:     容量问答（翻倍响应时间、瓶颈转移、最少台数）
- simulation: 离散事件仿真，作为解析模型的验收基准
- scenario:   不动产登记系统示例场景
"""

from .network import Station, Pool, Step, RequestClass, QueueNetwork, FCFS, PS, IS
from .mva import solve
from .layered import LayeredSolver
from .sizing import what_if, bottleneck_chain, min_servers_for_target
from . import simulation, estimation, burst, scenario, validate

__all__ = [
    "Station", "Pool", "Step", "RequestClass", "QueueNetwork",
    "FCFS", "PS", "IS", "solve", "LayeredSolver",
    "what_if", "bottleneck_chain", "min_servers_for_target",
    "simulation", "estimation", "burst", "scenario", "validate",
]
