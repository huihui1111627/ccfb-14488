"""零依赖 Web 应用: 排队网络可视化与交互式容量评估。

启动::

    python3 -m qncap.webapp [--port 8000]

接口:
* GET  /                可视化页面(原生 SVG + JS, 无外部库)
* GET  /api/network     网络拓扑(站点/边/类)
* GET  /api/solve?query=N&register=N&booking=R&bulk=R
                        调整用户数/到达率后的解析结果(利用率/响应/吞吐/瓶颈)
* GET  /api/sweep?var=query&lo=..&hi=..  单变量扫描曲线
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

from . import scenario, mva, sizing


def _network_payload(net) -> dict:
    layers = net.station_layers()
    stations = []
    for name, s in net.stations.items():
        stations.append({
            "name": name,
            "sched": s.sched,
            "servers": s.servers,
            "pool": s.pool_capacity,
            "layer": layers[name],
        })
    edges = [{"from": u, "to": v} for u, v in net.call_edges()]
    classes = []
    for c in net.classes.values():
        classes.append({
            "name": c.name, "kind": c.kind,
            "population": c.population,
            "arrival_rate": c.arrival_rate,
            "think_time": c.think_time,
            "color": c.color,
        })
    return {"name": net.name, "stations": stations,
            "edges": edges, "classes": classes}


def solve_payload(query: int, register: int,
                  booking: float, bulk: float) -> dict:
    net = scenario.build_flat_network(
        closed_pop={"query": query, "register": register},
        open_rates={"booking": booking, "bulk": bulk})
    res = mva.solve_approx(net)
    ranks = sizing.rank_bottlenecks(net, res)
    return {
        "stable": res.stable,
        "warnings": res.warnings,
        "stations": {
            s: {
                "utilization": res.station_utilization[s],
                "queue_length": res.station_queue_length[s],
                "residence": res.station_residence[s],
                "throughput": res.station_throughput[s],
                "servers": net.stations[s].servers,
                "sched": net.stations[s].sched,
            } for s in net.stations},
        "classes": {
            c: {
                "response_time": res.class_response[c],
                "throughput": res.class_throughput[c],
                "kind": net.classes[c].kind,
                "color": net.classes[c].color,
            } for c in net.classes},
        "bottleneck_order": [r.station for r in ranks
                             if net.stations[r.station].sched != "is"],
        "bottleneck_reasons": {r.station: r.reason for r in ranks},
    }


def sweep_payload(var: str, lo: float, hi: float, steps: int = 24) -> dict:
    grid = [lo + (hi - lo) * i / (steps - 1) for i in range(steps)]
    pts = []
    base = scenario.build_scenario("medium")
    for val in grid:
        net = scenario.build_flat_network(
            closed_pop={"query": base.classes["query"].population,
                        "register": base.classes["register"].population},
            open_rates={"booking": base.classes["booking"].arrival_rate,
                        "bulk": base.classes["bulk"].arrival_rate})
        if var in ("query", "register"):
            net.classes[var].population = max(1, int(round(val)))
        else:
            net.classes[var].arrival_rate = max(0.01, val)
        res = mva.solve_approx(net)
        pts.append({
            "x": round(val, 3),
            "response": {c: res.class_response[c] for c in net.classes},
            "util": {s: res.station_utilization[s] for s in net.stations},
            "stable": res.stable,
        })
    return {"variable": var, "points": pts}


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # 静默
        pass

    def _send(self, obj: dict, code: int = 200) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        q = parse_qs(parsed.query)
        try:
            if path == "/":
                self._send_page()
            elif path == "/api/network":
                self._send(_network_payload(
                    scenario.build_scenario("medium")))
            elif path == "/api/solve":
                self._send(solve_payload(
                    int(q.get("query", [60])[0]),
                    int(q.get("register", [24])[0]),
                    float(q.get("booking", [6])[0]),
                    float(q.get("bulk", [1.5])[0])))
            elif path == "/api/sweep":
                self._send(sweep_payload(
                    q.get("var", ["query"])[0],
                    float(q.get("lo", [10])[0]),
                    float(q.get("hi", [200])[0]),
                    int(q.get("steps", [24])[0])))
            else:
                self._send({"error": "not found"}, 404)
        except Exception as exc:  # noqa: BLE001
            self._send({"error": str(exc)}, 500)

    def _send_page(self) -> None:
        body = PAGE_HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    ap = argparse.ArgumentParser(description="排队网络容量评估可视化")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    srv = ThreadingHTTPServer((args.host, args.port), _Handler)
    print(f"可视化页面: http://{args.host}:{args.port}/")
    print("Ctrl+C 退出")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        srv.shutdown()


# 页面在 webapp_page.py 中, 保持本文件聚焦于接口
from .webapp_page import PAGE_HTML  # noqa: E402

if __name__ == "__main__":
    main()
