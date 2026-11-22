# 排队网络容量评估工具 (qncap)

面向不动产登记这类多层系统的容量规划工具。用**排队网络解析模型**做快速
容量评估, 用自包含的**离散事件仿真**充当"真实系统"来验收解析预测。
纯 Python + NumPy, 无其他第三方依赖; Web 页面用标准库 `http.server`
加原生 SVG/JS, 可完全离线运行。

## 要解决的问题

> 每年交税截止前扩容全凭去年经验; 去年应用服务器加了一倍, 结果卡在
> 数据库连接池上, 钱花了系统照样挂。

根因是两类传统容量方法都失效: 按平均利用率拍脑袋看不到排队和瓶颈转移;
只扩容应用层却忽略了**连接池/线程池这类有限资源的嵌套占用**——请求拿着
应用线程等数据库连接, 应用再多也会堵在池上。本工具用排队论把这些效应
显式建模。

## 排队网络模型

五个站点, 三种调度规则:

| 站点 | 调度 | 含义 |
|------|------|------|
| gateway 网关 | FCFS | 先到先服务, 多服务台 |
| app 应用 | PS | 处理器共享(线程池), 多服务台 |
| cache 缓存 | IS | 纯延迟(无限服务台, 命中即返回) |
| db 数据库 | FCFS | 多服务台 + 连接池 |
| ext 外部接口 | FCFS | 税务/住建共享接口, 多服务台 |

四类请求, 闭合类(固定用户反复操作)与开放类(外部持续到达)混合:

* `query` 查询、`register` 登簿: 闭合类, 带 think time;
* `booking` 网上预约、`bulk` 批量上报: 开放类, 泊松/非齐次泊松到达。

站点间通过**同步嵌套调用**连接(应用线程在等数据库返回期间不释放),
请求在各站点有独立的服务需求与调用次数(支持 0.6 这类概率调用)。

## 模块结构

```
qncap/
├── network.py     排队网络数据模型(站点/类/嵌套调用/需求展开)
├── sim.py         离散事件仿真: FCFS/PS/IS、资源池嵌套占用、开闭混合、NHPP
├── mva.py         精确多链 MVA + Bard-Schweitzer 近似迭代 MVA + Erlang-C
├── layered.py     分层模型: 线程池/连接池的 FES 与 Erlang-B 阻塞修正
├── estimation.py  从监控利用率/吞吐反推服务需求(自实现 NNLS)
├── burst.py       非平稳/突发修正(Allen-Cunneen G/G/m + 峰值等效负载)
├── sizing.py      瓶颈识别、瓶颈转移、最小台数规划、what-if
├── scenario.py    不动产登记系统参考场景(轻/中/重三档负载)
├── validation.py  解析 vs 仿真三档负载验收
├── webapp.py      零依赖 Web 服务(JSON API)
└── webapp_page.py 排队网络图 + 滑块交互页面
```

## 快速开始

```bash
# 1. 跑解析 vs 仿真验收(数分钟, 输出三档负载偏差表 + 瓶颈转移 + 容量规划)
python3 -m qncap.validation
python3 -m qncap.validation --quick     # 快速冒烟

# 2. 启动可视化页面
python3 -m qncap.webapp --port 8000
# 浏览器打开 http://127.0.0.1:8000/ , 拖动用户数滑块看利用率/响应时间
```

## 核心算法

### 精确 MVA (mva.solve_exact)
多闭合链平均值分析, 按人口向量格点递归, 每个多服务台站点维护边际占用
分布(Reiser load-dependent 递归)。复杂度随人口格点数增长, 用户数大时慢。

### 近似迭代 MVA (mva.solve_approx)
Bard-Schweiter 比例缩放迭代, 复杂度与用户数无关, 支持大人口与混合开闭
网络。单服务台用到达定理精确缩放; 多服务台用 M/M/m (Erlang-C) 求站点
平均等待, 按各类访问次数计价。开放类作为背景负载并入。

### 分层资源池 (layered.solve_layered)
在正确展开的需求矩阵上跑标准 MVA, 再对每个带池站点用 Erlang-B 估计
令牌全占满的阻塞概率、Erlang-C 估计等待概率, 把"持上层资源等令牌"的
延迟按访问次数加到端到端响应。池容量与服务台数相互独立。

### 需求反推 (estimation.estimate_demands)
由利用率定律 `U_i * m_i = Σ_k X_k * D[k,i]`, 对每个站点解一个非负
最小二乘问题。NNLS (Lawson-Hanson active-set) 纯 NumPy 实现,
与 scipy 结果一致; 吞吐矩阵欠秩时给出可辨识性告警。

### 突发修正 (burst.solve_with_burst)
平稳 M/M/m 假设在尖峰流量下偏乐观。两层修正: Allen-Cunneen 变异系数
`W_GGm ≈ W_MMm·(Ca²+Cs²)/2`, 以及峰值区间等效到达率(peak factor)。

## 验收结论(quick 配置, 4 次独立重复仿真)

| 负载 | 利用率最大偏差 | 响应时间最大偏差 | 吞吐最大偏差 | 瓶颈预测 |
|------|--------------|----------------|------------|---------|
| 轻载 | 1.5% | 1.1% | 0.8% | ✓ ext 一致 |
| 中载 | 1.0% | 0.8% | 0.6% | ✓ ext 一致 |
| 重负载 | 1.2% | 6.7% | 1.8% | ✓ ext 一致 |

瓶颈转移序列(重负载, 逐次加一台): `ext → db → ext → db`,
与两站利用率交替领先一致。用户数翻倍后正确识别 ext 饱和并告警。

### 已知边界(如实说明)
* **接近饱和时**解析模型偏保守(预测响应时间高于仿真), 这是 M/M/m
  在高利用率下等待时间方差放大的固有特性, 方向安全(不会低估风险)。
* **深度池饱和**(连接池容量只有正常值 1/10 的极端场景)下, 分层模型
  能正确诊断 `P(block)` 并给出恶化方向与量级, 绝对值偏保守;
  硬饱和区的精确预测应以仿真为准。
* 解析模型假设服务时间可用均值/变异系数刻画, 长尾依赖(如外部接口
  级联超时)需要在仿真侧用对应 cv 的 Gamma 分布单独评估。

## API 一览

```python
from qncap import scenario, mva, sizing
from qncap.sim import run_replications
from qncap.layered import solve_layered

net = scenario.build_scenario("heavy")          # 或 build_layered_network
ana = mva.solve_approx(net)                     # 解析预测
sim = run_replications(net, n_rep=5)            # 仿真对照

# 用户数翻倍
doubled = sizing.what_if_population(
    net, mva.solve_approx, {"query": 2.0, "register": 2.0})
# 瓶颈转移顺序
seq = sizing.bottleneck_shift_sequence(net, mva.solve_approx)
# 达到响应目标的最小台数(自动校验目标是否低于物理下限)
config, result, notes = sizing.min_servers_for_target(net, mva.solve_approx, target)
```
