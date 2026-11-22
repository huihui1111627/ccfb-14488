"""基于排队网络的容量评估工具包。

模块组成::

    network     排队网络数据模型(站点/请求类/嵌套调用)
    sim         离散事件仿真(被评估系统的替代环境)
    mva         精确 MVA 与 Bard-Schweitzer 近似迭代 MVA
    layered     分层模型(流等效服务站 FES,处理线程池/连接池嵌套占用)
    estimation  由监控利用率与吞吐反推各类请求的服务需求
    burst       突发/非平稳到达修正
    sizing      瓶颈分析、瓶颈转移、最小配置规划
    scenario    不动产登记系统参考场景
    validation  解析模型与仿真的三档负载对比验收
"""

__all__ = [
    "network",
    "sim",
    "mva",
    "layered",
    "estimation",
    "burst",
    "sizing",
    "scenario",
    "validation",
]
