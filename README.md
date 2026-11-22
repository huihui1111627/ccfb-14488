# 不动产登记系统 · 排队网络容量评估工具

基于排队论的容量规划工具：用**解析模型**（平均值分析 MVA + 分层 Erlang-C）
快速预测扩容效果，用**离散事件仿真**作为验收基准。纯 Python + NumPy 实现，
Web 界面仅用标准库，内网可离线运行。

## 解决的问题

交税截止日前扩容全凭经验——去年把应用服务器加了一倍，却卡在数据库连接池上，
钱花了系统照样挂。本工具把系统建成排队网络，量化回答：

- 用户数翻倍后各类请求响应时间到多少？瓶颈在哪个站点/资源池？
- 瓶颈解掉以后下一个瓶颈在哪（瓶颈转移链）？
- 要达到给定响应时间，每个站点/连接池最少配多少？
- 连接池、线程池这类"拿一层资源等下一层"的嵌套占用如何影响容量？

## 排队网络模型

```
用户/外部系统
    │
[网关 gateway]  FCFS ×4   鉴权、限流
    │
[应用 app]      PS  ×8   CPU 处理器共享，受 app_threads(24) 线程池约束
    │
    ├─ [缓存 cache]  FCFS ×2
    ├─ [数据库 db]   FCFS ×8，受 db_connections(10) 连接池约束
    └─ [外部接口 ext] IS 纯延迟（无限服务台，不排队）
```

四类请求：`register` 登记发证、`query` 查询核验、`tax` 税务联办（均为固定
用户数的闭合类，带思考时间），`push` 第三方推送（泊松/MMPP 开放到达）。

## 核心算法

| 需求 | 实现 |
|------|------|
| 闭合网络求解 | 精确多类 MVA（人口格点递推，m=1 严格精确）|
| 大人口性能 | Schweitzer-Bard 近似 MVA 固定点迭代，O(迭代×C×S) |
| 开放网络/资源池 | 多服务台 Erlang-C，GI/GI/m 的 Allen-Cunneen 修正 |
| 嵌套资源占用 | 分层模型：池等价 M/M/K，等待并入闭合类等效思考时间做不动点 |
| 服务需求反推 | 多窗口利用率/吞吐的非负最小二乘（自实现 Lawson-Hanson NNLS）|
| 突发/非平稳 | 变异系数修正、峰值区间、MMPP-2 标定与等效 ca² |
| 验收基准 | 离散事件仿真：精确 PS 份额调度、信号量嵌套、批均值 95% 置信区间 |

## 目录结构

```
captool/
  network.py     站点/资源池/请求类/路由定义
  mva.py         Erlang-C、精确 MVA、Schweitzer 近似
  layered.py     分层排队模型（连接池/线程池嵌套）
  estimation.py  服务需求反推（NNLS）
  burst.py       突发与非平稳修正
  sizing.py      翻倍预测、瓶颈转移链、最少台数规划
  simulation.py  离散事件仿真
  scenario.py    不动产登记示例场景与三档负载
  validate.py    三档负载解析 vs 仿真验收
  cli.py         命令行入口
web/
  app.py         标准库 HTTP 服务 + JSON API
  index.html     单页 SVG 网络图与交互滑块
tests/
  test_core.py   公式级单元测试
```

## 使用

```bash
# 单元测试
PYTHONPATH=. python3 tests/test_core.py

# 求解某档负载（light / medium / heavy）
PYTHONPATH=. python3 -m captool.cli solve --level heavy

# 瓶颈转移链
PYTHONPATH=. python3 -m captool.cli chain --level heavy

# 用户数翻倍 what-if
PYTHONPATH=. python3 -m captool.cli double

# 从合成监控反推服务需求
PYTHONPATH=. python3 -m captool.cli estimate

# 三档负载解析 vs 仿真验收
PYTHONPATH=. python3 -m captool.cli validate --seconds 250 --rep 2

# 启动可视化 Web（http://127.0.0.1:8000）
PYTHONPATH=. python3 -m captool.cli web
```

## 验收结论（示例场景）

三档负载下解析模型与离散事件仿真对比，17 项指标（各类响应时间、吞吐、
各站点利用率、资源池利用率与等待）全部落在工程阈值内：

- 轻载/中载：响应时间与利用率偏差 < 8%
- 近饱和：偏差 < 9%，首要瓶颈预测一致
- 瓶颈转移链解析与仿真**完全一致**：
  `db → gateway → app → db_connections → app_threads`
- 翻倍后首要瓶颈从数据库服务器**转移到数据库连接池**——量化复现了
  "只扩应用服务器却卡在连接池"的失败模式。

## 依赖

仅 NumPy（运行）；Web 与全部算法不依赖任何第三方 Web 框架或 SciPy。
Python ≥ 3.9。
