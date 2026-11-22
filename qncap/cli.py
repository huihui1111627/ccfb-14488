"""命令行入口: 快速求解指定负载档位。

用法::

    python3 -m qncap.cli --level heavy
    python3 -m qncap.cli --level heavy --layered
    python3 -m qncap.cli --query 120 --register 48 --booking 11 --bulk 3
"""

from __future__ import annotations

import argparse

from . import scenario, mva, sizing
from .layered import solve_layered


def print_result(net, res, title: str) -> None:
    print("=" * 70)
    print(title)
    print("=" * 70)
    print("\n[站点]")
    print(f"  {'站点':<9}{'调度':<6}{'台/池':<8}{'利用率':>8}{'队列长':>9}{'驻留(s)':>10}")
    for name, s in net.stations.items():
        cap = f"{s.servers or '∞'}" + (f"/{s.pool_capacity}"
                                        if s.pool_capacity else "")
        print(f"  {name:<9}{s.sched:<6}{cap:<8}"
              f"{res.station_utilization[name]:>8.3f}"
              f"{res.station_queue_length[name]:>9.2f}"
              f"{res.station_residence[name]:>10.4f}")
    print("\n[请求类]")
    print(f"  {'类':<10}{'类型':<7}{'响应(s)':>10}{'吞吐(tps)':>11}")
    for cname, c in net.classes.items():
        print(f"  {cname:<10}{c.kind:<7}"
              f"{res.class_response[cname]:>10.4f}"
              f"{res.class_throughput[cname]:>11.3f}")
    ranks = sizing.rank_bottlenecks(net, res)
    chain = [r.station for r in ranks
             if net.stations[r.station].sched != "is"]
    print("\n瓶颈优先级: " + " → ".join(chain))
    if res.warnings:
        print("告警: " + "; ".join(res.warnings))


def main() -> None:
    ap = argparse.ArgumentParser(description="排队网络容量评估 CLI")
    ap.add_argument("--level", choices=["light", "medium", "heavy"],
                    default="medium")
    ap.add_argument("--layered", action="store_true",
                    help="使用带线程池/连接池的分层模型")
    ap.add_argument("--query", type=int)
    ap.add_argument("--register", type=int)
    ap.add_argument("--booking", type=float)
    ap.add_argument("--bulk", type=float)
    args = ap.parse_args()

    if args.layered:
        net = scenario.build_layered_network(
            scenario.LOAD_LEVELS[args.level],
            scenario.OPEN_LEVELS[args.level])
    else:
        net = scenario.build_scenario(args.level)
    if args.query is not None:
        net.classes["query"].population = args.query
    if args.register is not None:
        net.classes["register"].population = args.register
    if args.booking is not None:
        net.classes["booking"].arrival_rate = args.booking
    if args.bulk is not None:
        net.classes["bulk"].arrival_rate = args.bulk

    res = solve_layered(net) if args.layered else mva.solve_approx(net)
    title = f"{net.name} · 负载={args.level}" + (" · 分层池化" if args.layered else "")
    print_result(net, res, title)


if __name__ == "__main__":
    main()
