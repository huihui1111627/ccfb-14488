# -*- coding: utf-8 -*-
"""容量评估 Web 应用（仅依赖 Python 标准库，内网可离线运行）。

启动：python3 -m captool.cli web   或   python3 web/app.py
访问：http://127.0.0.1:8000

页面功能
========
- SVG 绘制排队网络：网关/应用/缓存/数据库/外部接口 + 两个资源池；
- 拖动用户数滑块，实时重算各站点利用率、各资源池占用与各类响应时间；
- 展示瓶颈转移链、用户翻倍 what-if、突发修正开关；
- 所有计算在后端 :mod:`captool` 完成，前端只负责渲染。
"""
from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

from captool import scenario, sizing, burst
from captool.layered import LayeredSolver
from captool.network import IS

_HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------- #
# 计算 API
# ---------------------------------------------------------------------- #
def _serialize_solution(net, sol, pop_scale, arr_scale):
    stations = []
    for si, st in enumerate(net.stations):
        stations.append({
            "name": st.name, "discipline": st.discipline,
            "servers": st.servers,
            "util": round(float(sol.util[si]), 4),
            "queue": round(float(sol.Lq[si]), 3),
            "pwait": round(float(sol.pwait[si]), 4),
            "is_delay": st.discipline == IS,
        })
    pools = [{
        "name": pname,
        "capacity": int(sol.pool_servers[k]),
        "util": round(float(sol.pool_util[k]), 4),
        "wait": round(float(sol.pool_wait[k]), 4),
        "queue": round(float(sol.pool_queue[k]), 3),
    } for k, pname in enumerate(sol.pool_names)]
    classes = [{
        "name": net.classes[ci].name,
        "kind": net.classes[ci].kind,
        "rt": round(float(sol.R[ci]), 4),
        "x": round(float(sol.X[ci]), 3),
        "population": net.classes[ci].population,
        "arrival": net.classes[ci].arrival,
    } for ci in range(net.C)]
    bn = sizing.bottleneck(sol)
    return {
        "stations": stations, "pools": pools, "classes": classes,
        "stable": sol.stable, "converged": sol.converged,
        "bottleneck": bn["name"],
        "pop_scale": pop_scale, "arr_scale": arr_scale,
    }


def evaluate(pop_scale: float, arr_scale: float, burst_cv: float = 1.0):
    net = scenario.build_network(population_scale=pop_scale,
                                 arrival_scale=arr_scale)
    cv = None
    if burst_cv > 1.0:
        cv = {st.name: (burst_cv, 1.0)
              for st in net.stations if st.discipline != IS}
    sol = LayeredSolver(net).solve(cv=cv)
    return net, sol


def api_state(params) -> dict:
    pop = float(params.get("pop", ["1.0"])[0])
    arr = float(params.get("arr", ["1.0"])[0])
    cv = float(params.get("cv", ["1.0"])[0])
    net, sol = evaluate(pop, arr, cv)
    data = _serialize_solution(net, sol, pop, arr)
    # 翻倍 what-if
    net2, sol2 = evaluate(pop * 2.0, arr * 2.0, cv)
    data["double"] = {
        "classes": [{
            "name": net2.classes[ci].name,
            "rt": round(float(sol2.R[ci]), 4),
            "ratio": round(float(sol2.R[ci] / max(sol.R[ci], 1e-9)), 2),
        } for ci in range(net.C)],
        "bottleneck": sizing.bottleneck(sol2)["name"],
        "stable": sol2.stable,
    }
    # 瓶颈转移链
    chain = sizing.bottleneck_chain(net, max_steps=6, util_headroom=0.45)
    data["chain"] = [{"order": s.order, "name": s.name,
                      "kind": s.kind, "util": round(s.util, 4),
                      "action": s.action} for s in chain.chain]
    data["topology"] = _topology(net)
    return data


def _topology(net) -> dict:
    """提取网络拓扑（站点坐标与边）供 SVG 渲染。"""
    pos = {
        "gateway": (90, 200), "app": (280, 200), "cache": (470, 90),
        "db": (470, 200), "ext": (470, 320),
        "app_threads": (280, 70), "db_connections": (470, 300),
    }
    edges = []
    seen = set()
    for c in net.classes:
        for step in c.route:
            pass
    # 依据路由生成站点间有向边（去重）
    names = [s.name for s in net.stations]
    for c in net.classes:
        seq = [st.station for st in c.route]
        for a, b in zip(seq, seq[1:]):
            if a != b and (a, b) not in seen:
                seen.add((a, b))
                edges.append({"from": a, "to": b})
    return {"positions": pos, "edges": edges,
            "stations": names,
            "pools": [p.name for p in net.pools]}


# ---------------------------------------------------------------------- #
# HTTP 服务
# ---------------------------------------------------------------------- #
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, code, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            with open(os.path.join(_HERE, "index.html"), "rb") as f:
                self._send(200, f.read(), "text/html; charset=utf-8")
        elif parsed.path == "/api/state":
            params = parse_qs(parsed.query)
            try:
                data = api_state(params)
                self._send(200, json.dumps(data).encode("utf-8"),
                           "application/json")
            except Exception as exc:  # 页面不能因单点计算失败白屏
                self._send(500, json.dumps(
                    {"error": str(exc)}).encode("utf-8"),
                    "application/json")
        else:
            self._send(404, b"not found", "text/plain")


def main(port: int = 8000):
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"容量评估 Web 已启动: http://127.0.0.1:{port}")
    print("按 Ctrl+C 停止")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
