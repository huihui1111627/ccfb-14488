# -*- coding: utf-8 -*-
"""命令行入口：解析求解、仿真验收、需求反推演示、启动 Web。

用法
====
python3 -m captool.cli solve [--level light|medium|heavy]
python3 -m captool.cli validate [--seconds 300] [--rep 3]
python3 -m captool.cli estimate
python3 -m captool.cli chain [--level heavy]
python3 -m captool.cli double
python3 -m captool.cli web [--port 8000]
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from . import scenario, sizing, estimation
from .layered import LayeredSolver


def cmd_solve(args):
    net = scenario.load_network(args.level)
    sol = LayeredSolver(net).solve()
    print(sol.summary())


def cmd_chain(args):
    net = scenario.load_network(args.level)
    print(sizing.bottleneck_chain(
        net, max_steps=args.steps, util_headroom=args.headroom).summary())


def cmd_double(args):
    net = scenario.build_network()
    print(sizing.what_if(net, population_scale=2.0,
                         arrival_scale=2.0).summary())


def cmd_estimate(args):
    net = scenario.build_network()
    samples = scenario.synthetic_monitor(net, n_windows=args.windows)
    est = estimation.infer_demands(net, samples)
    print(est.summary(net))
    enet = estimation.replace_network_demands(net, est)
    s1 = LayeredSolver(net).solve()
    s2 = LayeredSolver(enet).solve()
    print("\n用反推需求重解的端到端响应时间 vs 原网络:")
    for ci, c in enumerate(net.classes):
        print(f"  {c.name:<10} 原={s1.R[ci]:.4f}s  反推={s2.R[ci]:.4f}s  "
              f"偏差={abs(s2.R[ci]-s1.R[ci])/s1.R[ci]:.1%}")


def cmd_validate(args):
    from .validate import run_validation
    rep = run_validation(sim_seconds=args.seconds, n_rep=args.rep,
                         min_completed=args.min_completed)
    print(rep.summary())


def cmd_web(args):
    import os
    sys.path.insert(0, os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
    web_dir = os.path.join(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))), "web")
    sys.path.insert(0, web_dir)
    from app import main
    main(args.port)


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="captool", description="排队网络容量评估工具")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("solve", help="求解指定负载档位")
    sp.add_argument("--level", choices=list(scenario.LOAD_LEVELS),
                    default="medium")
    sp.set_defaults(func=cmd_solve)

    sp = sub.add_parser("chain", help="瓶颈转移链")
    sp.add_argument("--level", choices=list(scenario.LOAD_LEVELS),
                    default="heavy")
    sp.add_argument("--steps", type=int, default=6)
    sp.add_argument("--headroom", type=float, default=0.45)
    sp.set_defaults(func=cmd_chain)

    sp = sub.add_parser("double", help="用户数翻倍 what-if")
    sp.set_defaults(func=cmd_double)

    sp = sub.add_parser("estimate", help="服务需求反推演示")
    sp.add_argument("--windows", type=int, default=30)
    sp.set_defaults(func=cmd_estimate)

    sp = sub.add_parser("validate", help="解析 vs 仿真三档验收")
    sp.add_argument("--seconds", type=float, default=300)
    sp.add_argument("--rep", type=int, default=2)
    sp.add_argument("--min-completed", type=int, default=10000)
    sp.set_defaults(func=cmd_validate)

    sp = sub.add_parser("web", help="启动可视化 Web")
    sp.add_argument("--port", type=int, default=8000)
    sp.set_defaults(func=cmd_web)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
