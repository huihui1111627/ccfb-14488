# -*- coding: utf-8 -*-
"""公式级单元测试：不依赖长仿真，快速验证解析算法的正确性。

运行：python3 -m pytest tests/ -q   或   python3 tests/test_core.py
"""
import math
import unittest

import numpy as np

from captool.network import (Station, Pool, Step, RequestClass,
                             QueueNetwork, FCFS, PS, IS)
from captool.mva import erlang_c, queue_factor, solve
from captool.estimation import nnls, infer_demands, MonitorSample
from captool import burst


def single_station(disc, m=1, S=0.1, N=20, Z=2.0):
    st = [Station("s", m, disc)]
    cls = [RequestClass("c", "closed", population=N, think=Z,
                        route=[Step("s", S)])]
    return QueueNetwork(st, cls)


class TestErlangC(unittest.TestCase):
    def test_mm1_equals_rho(self):
        # M/M/1 时 Erlang-C 等待概率 = 利用率
        for rho in (0.1, 0.5, 0.9, 0.95):
            self.assertAlmostEqual(erlang_c(rho, 1), rho, places=6)

    def test_zero_load(self):
        self.assertEqual(erlang_c(0.0, 4), 0.0)

    def test_known_mm2(self):
        # a=1.6, m=2 的手算值
        p = erlang_c(1.6, 2)
        self.assertGreater(p, 0.0)
        self.assertLess(p, 1.0)
        self.assertAlmostEqual(p, 0.7111, places=3)

    def test_saturating_does_not_explode(self):
        # 到达定理口径允许 a>=m，因子必须有限
        f, rho, pw = queue_factor(3.0, 1)
        self.assertTrue(math.isfinite(f))
        self.assertAlmostEqual(f, 1.0 + 3.0, places=6)   # m=1 → 1+a
        f2, _, _ = queue_factor(5.0, 2)
        self.assertTrue(math.isfinite(f2))


class TestExactMVA(unittest.TestCase):
    def test_mm1_closed_matches_manual_recurrence(self):
        S, Z, N = 0.1, 2.0, 20
        L = 0.0
        for n in range(1, N + 1):
            R = S * (1 + L)
            X = n / (Z + R)
            L = X * R
        sol = solve(single_station(FCFS, 1, S, N, Z))
        self.assertEqual(sol.engine, "exact-mva")
        self.assertAlmostEqual(sol.R[0], R, places=4)
        self.assertAlmostEqual(sol.X[0], X, places=4)
        self.assertAlmostEqual(sol.util[0], X * S, places=3)

    def test_is_station_response_equals_service(self):
        # IS 纯延迟站不排队，R 恒等于服务时间
        sol = solve(single_station(IS, 1, 0.3, N=30, Z=1.0))
        self.assertAlmostEqual(sol.R[0], 0.3, places=3)

    def test_throughput_conservation(self):
        # 闭合网络 Little 定律：N = X(R+Z)
        net = single_station(FCFS, 2, 0.05, N=40, Z=1.5)
        sol = solve(net)
        self.assertAlmostEqual(sol.X[0] * (sol.R[0] + 1.5), 40, places=3)


class TestSchweitzerApprox(unittest.TestCase):
    def test_large_population_runs_and_finite(self):
        # 人口 400 触发近似路径，结果须有限且接近饱和
        net = single_station(FCFS, 4, 0.2, N=400, Z=2.0)
        sol = solve(net)
        self.assertTrue(np.all(np.isfinite(sol.R)))
        self.assertTrue(np.all(np.isfinite(sol.X)))
        self.assertGreater(sol.util[0], 0.9)

    def test_approx_close_to_exact_moderate(self):
        # 中载（利用率 70%~80%）下近似与精确解偏差应在 12% 内；
        # 饱和点附近 Schweitzer 启发式偏差会放大，属已知特性。
        net = single_station(FCFS, 3, 0.04, N=40, Z=2.0)
        exact = solve(net, approx="exact")
        approx = solve(net, approx="schweitzer")
        self.assertLess(exact.util[0], 0.85)
        self.assertLess(abs(exact.R[0] - approx.R[0]) / exact.R[0], 0.12)


class TestNNLS(unittest.TestCase):
    def test_recovers_known_nonnegative(self):
        rng = np.random.default_rng(0)
        A = rng.uniform(0, 5, size=(20, 3))
        x_true = np.array([2.0, 0.0, 1.5])
        b = A @ x_true
        x = nnls(A, b)
        np.testing.assert_allclose(x, x_true, atol=1e-6)

    def test_nonnegativity(self):
        rng = np.random.default_rng(1)
        A = rng.normal(size=(15, 4))
        b = rng.normal(size=15)
        x = nnls(A, b)
        self.assertTrue(np.all(x >= -1e-12))


class TestDemandInference(unittest.TestCase):
    def test_recovers_demands_from_noiseless_law(self):
        # 构造良态的无噪声服务定律观测，反推应还原真实需求
        stations = [Station("a", 2, FCFS), Station("b", 1, FCFS)]
        c1 = RequestClass("c1", "open", arrival=1.0,
                          route=[Step("a", 0.1), Step("b", 0.2)])
        c2 = RequestClass("c2", "open", arrival=1.0,
                          route=[Step("a", 0.3)])
        net = QueueNetwork(stations, [c1, c2])
        D = net.demands
        samples = []
        rng = np.random.default_rng(2)
        for _ in range(12):
            scales = rng.uniform(0.5, 1.5, size=2)
            thr = {"c1": scales[0], "c2": scales[1]}
            x = np.array([scales[0], scales[1]])
            util = {}
            for si, st in enumerate(stations):
                util[st.name] = float(np.sum(x * D[:, si])) / st.servers
            samples.append(MonitorSample(thr, util))
        est = infer_demands(net, samples)
        np.testing.assert_allclose(est.demands, D, atol=0.02)


class TestBurst(unittest.TestCase):
    def test_cv_squared_exponential(self):
        x = np.random.default_rng(3).exponential(1.0, 8000)
        self.assertAlmostEqual(burst.cv_squared(x), 1.0, delta=0.1)

    def test_peak_detection(self):
        counts = np.array([10, 12, 9, 60, 58, 11], dtype=float)
        pr = burst.peak_interval_rate(counts, 60, smooth_windows=1)
        self.assertGreater(pr.peak_lambda, pr.mean_lambda)
        self.assertEqual(pr.peak_index, 3)

    def test_mmpp_burst_fraction(self):
        m = burst.MMPP2(5.0, 20.0, r01=0.3, r10=0.7)
        self.assertAlmostEqual(m.burst_fraction, 0.3, places=6)
        self.assertGreaterEqual(m.equivalent_ca2(60), 1.0)

    def test_allen_cunneen_markov(self):
        # ca²=cs²=1 时系数为 1，不修正
        self.assertAlmostEqual(burst.allen_cunneen_correction(1, 1), 1.0)


class TestPoolValidation(unittest.TestCase):
    def _net(self, cap):
        stations = [Station("gw", 2, FCFS), Station("db", 2, FCFS)]
        pools = [Pool("dbc", cap)]
        c = RequestClass("c", "closed", population=15, think=2.0,
                         route=[Step("gw", 0.02),
                                Step("db", 0.08, acquire="dbc",
                                     release="dbc")])
        return QueueNetwork(stations, [c], pools)

    def test_layered_stable_and_finite(self):
        from captool.layered import LayeredSolver
        sol = LayeredSolver(self._net(8)).solve()
        self.assertTrue(np.all(np.isfinite(sol.R)))
        self.assertTrue(sol.converged)
        self.assertEqual(len(sol.pool_names), 1)
        self.assertGreaterEqual(sol.pool_util[0], 0.0)

    def test_small_pool_slower_than_large_pool(self):
        # 容量更小的连接池应导致不更优（RT 不更小）
        from captool.layered import LayeredSolver
        r_small = LayeredSolver(self._net(2)).solve()
        r_large = LayeredSolver(self._net(12)).solve()
        self.assertGreaterEqual(r_small.R[0], r_large.R[0] * 0.99)


if __name__ == "__main__":
    unittest.main(verbosity=2)
