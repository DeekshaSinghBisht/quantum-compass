"""Test suite for the Quantum Compass core engine.

Run with::

    pytest -q            # from the project root
    python -m unittest discover -s tests

The tests are deliberately small (few qubits, coarse grids) so the whole suite
runs in a few seconds while still validating the physics:

* graph generation + brute-force MaxCut ground truth,
* QAOA circuit structure (gate counts, angles, measurements),
* the exact evaluator vs the sampling evaluator (they must agree),
* the noise slider actually degrades the ratio (and matches sampling),
* landscape shape + local-extrema detection,
* the optimizer improves on a random start.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core.graph_generator import (  # noqa: E402
    compute_graph_properties,
    cut_value,
    generate_graph,
    optimal_maxcut,
)
from core.landscape import (  # noqa: E402
    compute_landscape,
    detect_local_maxima,
    detect_local_minima,
)
from core.noise_models import (  # noqa: E402
    ENVIRONMENTS,
    REPRESENTATIVE_LEVELS,
    SPACE_ENVIRONMENTS,
    effective_params,
    eta_from_elevation,
    get_environment_info,
    get_noise_model,
    noise_summary,
)
from core.optimizer import run_optimization  # noqa: E402
from core.qaoa_builder import build_qaoa_circuit  # noqa: E402
from core.qaoa_evaluator import (  # noqa: E402
    cost_operator,
    evaluate_qaoa,
    evaluate_qaoa_exact,
)
from core.classical_baselines import (  # noqa: E402
    GW_GUARANTEE,
    all_baselines,
    best_classical_ratio,
    goemans_willison_ratio,
    random_sampling_ratio,
    simulated_annealing_maxcut,
)
from core.advantage_map import (  # noqa: E402
    MAP_ENVIRONMENTS,
    MEMORY_MAX_N,
    compute_advantage_map,
    estimate_map_cost,
    reference_angles,
    summarize_advantage,
    valid_sizes,
)
from core.barren_plateau import (  # noqa: E402
    DEFAULT_EXACT_NOISE_MAX_N,
    EXACT_NOISE_MEMORY_MAX_N,
    classify_trainability,
    compare_diagrams,
    compute_barren_diagram,
    compute_trainability_score,
    diagram_depths,
    diagram_sizes,
    estimate_barren_cost,
    landscape_metrics,
)
from core.hardware_profiles import (  # noqa: E402
    AVAILABLE_HARDWARE,
    extract_hardware_stats,
    get_hardware_noise_model,
    hardware_noise_spec,
)
from core.mission_planner import generate_mission_plan, get_mission_template  # noqa: E402
from core.fingerprint import (  # noqa: E402
    DEFAULT_FINGERPRINT_ENVIRONMENTS,
    FINGERPRINT_FEATURE_NAMES,
    extract_landscape_fingerprint,
    generate_mystery_landscape,
    load_default_classifier,
    mystery_landscape_catalog,
    predict_environment,
    train_fingerprint_classifier,
)
from core.experimental_validation import (  # noqa: E402
    ExperimentalValidationError,
    analyze_validation_study,
    bootstrap_ratio_interval,
    create_experiment_manifest,
    distribution_metrics,
    manifest_to_json,
    normalize_bitstring,
    parse_counts_csv,
    parse_experiment_manifest,
    validation_report_markdown,
)


class TestGraphGenerator(unittest.TestCase):
    def test_three_regular_structure(self):
        G = generate_graph("3-regular", 6, seed=42)
        self.assertEqual(G.number_of_nodes(), 6)
        self.assertTrue(all(d == 3 for _, d in G.degree()))

    def test_erdos_renyi_and_scale_free(self):
        Ger = generate_graph("erdos-renyi", 8, seed=1, p=0.5)
        Gsf = generate_graph("scale-free", 8, seed=1, m=2)
        self.assertEqual(Ger.number_of_nodes(), 8)
        self.assertEqual(Gsf.number_of_nodes(), 8)
        self.assertGreater(Gsf.number_of_edges(), 0)

    def test_seeded_reproducibility(self):
        A = generate_graph("erdos-renyi", 8, seed=7, p=0.4)
        B = generate_graph("erdos-renyi", 8, seed=7, p=0.4)
        self.assertEqual(sorted(A.edges()), sorted(B.edges()))

    def test_invalid_family_raises(self):
        with self.assertRaises(ValueError):
            generate_graph("not-a-family", 6)

    def test_three_regular_odd_n_raises(self):
        with self.assertRaises(ValueError):
            generate_graph("3-regular", 7)  # 3*7 is odd → impossible

    def test_optimal_maxcut_known_triangle_plus(self):
        # A 4-cycle has MaxCut = 4 (perfect bipartition).
        import networkx as nx
        G = nx.cycle_graph(4)
        best, bits = optimal_maxcut(G)
        self.assertEqual(best, 4)
        self.assertEqual(len(bits), 4)
        self.assertEqual(cut_value([int(c) for c in bits], list(G.edges()), 4), 4)

    def test_optimal_maxcut_matches_brute_definition(self):
        G = generate_graph("3-regular", 6, seed=42)
        best, bits = optimal_maxcut(G)
        self.assertEqual(best, 7)  # known MaxCut of this instance
        self.assertEqual(cut_value([int(c) for c in bits], list(G.edges()), 6), best)

    def test_graph_properties_keys(self):
        G = generate_graph("3-regular", 6, seed=42)
        props = compute_graph_properties(G)
        for key in ("treewidth", "spectral_gap", "clustering_coeff", "density"):
            self.assertIn(key, props)
        self.assertAlmostEqual(props["density"], 9 / 15, places=6)  # 2|E|/(n(n-1))
        self.assertGreaterEqual(props["treewidth"], 1)


class TestQAOABuilder(unittest.TestCase):
    def test_circuit_structure_p1(self):
        G = generate_graph("3-regular", 6, seed=42)
        qc = build_qaoa_circuit(G, [0.4], [0.3])
        ops = dict(qc.count_ops())
        self.assertEqual(qc.num_qubits, 6)
        self.assertEqual(ops.get("h"), 6)              # one H per qubit
        self.assertEqual(ops.get("rzz"), 9)            # one RZZ per edge
        self.assertEqual(ops.get("rx"), 6)             # one RX per qubit
        self.assertEqual(ops.get("measure"), 6)        # measured
        self.assertEqual(qc.num_clbits, 6)

    def test_circuit_structure_p2(self):
        G = generate_graph("3-regular", 6, seed=42)
        qc = build_qaoa_circuit(G, [0.4, 0.2], [0.3, 0.1])
        ops = dict(qc.count_ops())
        self.assertEqual(ops.get("rzz"), 18)           # 2 layers × 9 edges
        self.assertEqual(ops.get("rx"), 12)            # 2 layers × 6 qubits
        self.assertEqual(ops.get("h"), 6)              # H only at the start

    def test_rzz_angle_is_twice_gamma(self):
        import networkx as nx
        G = nx.path_graph(2)  # single edge (0,1)
        qc = build_qaoa_circuit(G, [0.37], [0.0], measure=False)
        rzz = [instr for instr in qc.data if instr.operation.name == "rzz"][0]
        self.assertAlmostEqual(float(rzz.operation.params[0]), 2 * 0.37, places=10)

    def test_mismatched_angles_raise(self):
        G = generate_graph("3-regular", 6, seed=42)
        with self.assertRaises(ValueError):
            build_qaoa_circuit(G, [0.1, 0.2], [0.3])

    def test_no_measure_option(self):
        G = generate_graph("3-regular", 6, seed=42)
        qc = build_qaoa_circuit(G, [0.4], [0.3], measure=False)
        self.assertEqual(qc.num_clbits, 0)
        self.assertNotIn("measure", dict(qc.count_ops()))


class TestNoiseModels(unittest.TestCase):
    def test_ideal_is_none(self):
        self.assertIsNone(get_noise_model("ideal"))

    def test_earth_lab_builds_model(self):
        nm = get_noise_model("earth-lab", {"noise_level": 0.01})
        self.assertIsNotNone(nm)
        self.assertIn("rzz", nm.noise_instructions)
        self.assertIn("measure", nm.noise_instructions)

    def test_readout_prob_tagged_and_recovered(self):
        from core.qaoa_evaluator import _readout_prob_from_model
        nm = get_noise_model("earth-lab", {"noise_level": 0.03})
        self.assertAlmostEqual(_readout_prob_from_model(nm), 0.03, places=10)

    def test_noise_summary(self):
        s = noise_summary("earth-lab", {"noise_level": 0.02})
        self.assertAlmostEqual(s["p1"], 0.02, places=10)
        self.assertAlmostEqual(s["p2"], 0.02, places=10)
        self.assertAlmostEqual(s["readout"], 0.02, places=10)

    def test_unknown_environment_raises(self):
        with self.assertRaises(ValueError):
            get_noise_model("mars-lab")


class TestEvaluator(unittest.TestCase):
    def setUp(self):
        self.G = generate_graph("3-regular", 6, seed=42)
        self.opt, _ = optimal_maxcut(self.G)

    def test_cost_operator_matches_brute_force(self):
        # <C> on a computational basis state equals its cut value.
        from qiskit.quantum_info import Statevector
        from qiskit import QuantumCircuit
        op = cost_operator(self.G)
        n = 6
        for k in [0, 5, 21, 63]:
            qc = QuantumCircuit(n)
            for i in range(n):
                if (k >> i) & 1:
                    qc.x(i)
            sv = Statevector(qc)
            exp = float(np.real(sv.expectation_value(op)))
            bits = [(k >> i) & 1 for i in range(n)]
            self.assertAlmostEqual(exp, cut_value(bits, list(self.G.edges()), n), places=8)

    def test_ideal_exact_vs_sampling_agree(self):
        g, b = 0.285, 1.22
        ex = evaluate_qaoa_exact(self.G, [g], [b], noise_model=None, shots=8192)
        sa = evaluate_qaoa(self.G, [g], [b], noise_model=None, shots=20000)
        self.assertAlmostEqual(ex["approximation_ratio"], sa["approximation_ratio"], delta=0.02)
        self.assertEqual(ex["optimal_cut"], self.opt)
        self.assertEqual(len(ex["best_bitstring"]), 6)

    def test_ratio_bounds(self):
        ex = evaluate_qaoa_exact(self.G, [0.3], [0.4], noise_model=None, shots=1024)
        self.assertGreaterEqual(ex["approximation_ratio"], 0.0)
        self.assertLessEqual(ex["approximation_ratio"], 1.0 + 1e-9)

    def test_noise_degrades_ratio_and_matches_sampling(self):
        g, b = 0.285, 1.22
        ideal = evaluate_qaoa_exact(self.G, [g], [b], noise_model=None, shots=4096)
        nm = get_noise_model("earth-lab", {"noise_level": 0.06})
        noisy_ex = evaluate_qaoa_exact(self.G, [g], [b], noise_model=nm, shots=4096)
        noisy_sa = evaluate_qaoa(self.G, [g], [b], noise_model=nm, shots=40000)
        # noise should reduce the ratio at a good angle …
        self.assertLess(noisy_ex["approximation_ratio"], ideal["approximation_ratio"])
        # … and the exact path should agree with heavy sampling.
        self.assertAlmostEqual(
            noisy_ex["approximation_ratio"], noisy_sa["approximation_ratio"], delta=0.02
        )

    def test_counts_sum_to_shots(self):
        sa = evaluate_qaoa(self.G, [0.3], [0.4], noise_model=None, shots=1234)
        self.assertEqual(sum(sa["counts"].values()), 1234)

    def test_sampling_counts_use_documented_vertex_bit_order(self):
        # Public counts must use character i = graph vertex i, matching the
        # exact evaluator and graph visualisation rather than Qiskit's reversed
        # classical-register display convention.
        import networkx as nx
        graph = nx.Graph([(0, 1), (0, 2), (0, 3), (1, 2)])
        sampled = evaluate_qaoa(graph, [0.41], [0.29], noise_model=None, shots=12000)
        manual_expected = sum(
            cut_value([int(bit) for bit in bits], list(graph.edges()), graph.number_of_nodes()) * count
            for bits, count in sampled["counts"].items()
        ) / 12000.0
        self.assertAlmostEqual(manual_expected, sampled["expected_cut"], delta=0.02)


class TestLandscape(unittest.TestCase):
    def setUp(self):
        self.G = generate_graph("3-regular", 6, seed=42)

    def test_shape_and_progress(self):
        calls = []
        L = compute_landscape(self.G, depth=1, resolution=8, noise_model=None,
                              shots=512, progress_callback=lambda d, t: calls.append((d, t)))
        self.assertEqual(L["ratios"].shape, (8, 8))
        self.assertEqual(len(L["gammas"]), 8)
        self.assertEqual(len(L["betas"]), 8)
        self.assertTrue(calls)                 # progress callback fired
        self.assertEqual(calls[-1][0], calls[-1][1])  # finished all points
        self.assertLessEqual(L["best_ratio"], 1.0 + 1e-9)

    def test_local_extrema_detection(self):
        # Synthetic surface with a known interior min and max.
        gam = np.linspace(0, 2, 7)
        bet = np.linspace(0, 2, 7)
        GG, BB = np.meshgrid(gam, bet, indexing="ij")
        surface = (GG - 1.0) ** 2 + (BB - 1.0) ** 2   # single interior min at (1,1)
        mins = detect_local_minima(surface, gam, bet)
        maxs = detect_local_maxima(surface, gam, bet)
        self.assertTrue(any(abs(m["gamma"] - 1.0) < 1e-9 and abs(m["beta"] - 1.0) < 1e-9
                            for m in mins))
        self.assertEqual(len(maxs), 0)  # a convex bowl has no interior max

    def test_gamma_period_sweep(self):
        L = compute_landscape(self.G, depth=1, resolution=6, noise_model=None, shots=256)
        self.assertGreaterEqual(L["gammas"].max(), np.pi)  # spans beyond pi


class TestOptimizer(unittest.TestCase):
    def setUp(self):
        self.G = generate_graph("3-regular", 6, seed=42)

    def test_optimization_improves_and_records(self):
        opt = run_optimization(self.G, depth=1, noise_model=None, method="COBYLA",
                               max_iter=80, n_restarts=3, seed=0)
        self.assertIn("best_params", opt)
        self.assertEqual(len(opt["best_params"]), 2)
        self.assertGreater(opt["best_ratio"], 0.5)     # beats a trivial assignment
        self.assertTrue(len(opt["trajectory"]) > 0)    # trajectory recorded
        self.assertEqual(len(opt["all_trajectories"]), 3)

    def test_best_ratio_at_least_grid(self):
        # The optimizer should reach (within tolerance) the best grid ratio.
        L = compute_landscape(self.G, depth=1, resolution=12, noise_model=None, shots=512)
        opt = run_optimization(self.G, depth=1, noise_model=None, method="Nelder-Mead",
                               max_iter=120, n_restarts=4, seed=1)
        self.assertGreaterEqual(opt["best_ratio"] + 1e-6, L["best_ratio"] - 0.02)

    def test_invalid_method_raises(self):
        with self.assertRaises(ValueError):
            run_optimization(self.G, method="not-a-method")

    def test_depth2_param_length(self):
        opt = run_optimization(self.G, depth=2, noise_model=None, method="COBYLA",
                               max_iter=40, n_restarts=2, seed=2)
        self.assertEqual(len(opt["best_params"]), 4)   # 2 gammas + 2 betas


class TestSpaceNoiseModels(unittest.TestCase):
    """Phase 2: the five space environments + metadata (verification step 4)."""

    def test_all_environments_build_valid_models(self):
        from qiskit_aer.noise import NoiseModel
        for env in ENVIRONMENTS:
            params = ({"p1": 0.004, "p2": 0.01, "readout": 0.02, "burst_rate": 0.02}
                      if env == "custom" else {"noise_level": REPRESENTATIVE_LEVELS[env]})
            nm = get_noise_model(env, params)
            if env == "ideal":
                self.assertIsNone(nm)
            else:
                self.assertIsInstance(nm, NoiseModel, f"{env} did not build a NoiseModel")
                self.assertIn("rzz", nm.noise_instructions, f"{env} missing rzz error")

    def test_space_environments_present(self):
        for env in ["leo-satellite", "satellite-ground", "meo-van-allen", "deep-space"]:
            self.assertIn(env, ENVIRONMENTS)
            self.assertIn(env, SPACE_ENVIRONMENTS)

    def test_leo_thermal_modulation(self):
        # p2 = 0.008 * (1 + 0.30 thermal) = 0.0104 at the representative level.
        ep = effective_params("leo-satellite", {"noise_level": 0.008})
        self.assertAlmostEqual(ep["p2"], 0.0104, places=6)
        self.assertAlmostEqual(ep["p1"], 0.0026, places=6)   # 0.002 * 1.3
        self.assertAlmostEqual(ep["burst_rate"], 0.01, places=6)
        self.assertAlmostEqual(ep["readout"], 0.02, places=6)

    def test_burst_rates_increase_with_altitude(self):
        leo = effective_params("leo-satellite", {"noise_level": 0.008})["burst_rate"]
        meo = effective_params("meo-van-allen", {"noise_level": 0.020})["burst_rate"]
        deep = effective_params("deep-space", {"noise_level": 0.050})["burst_rate"]
        self.assertAlmostEqual(leo, 0.01, places=6)
        self.assertAlmostEqual(meo, 0.03, places=6)
        self.assertAlmostEqual(deep, 0.05, places=6)
        self.assertLess(leo, meo)
        self.assertLess(meo, deep)

    def test_satellite_ground_elevation_drives_loss(self):
        # Lower elevation → longer slant path → more photon loss.
        high = eta_from_elevation(90.0, 0.6)
        mid = eta_from_elevation(45.0, 0.6)
        low = eta_from_elevation(15.0, 0.6)
        self.assertAlmostEqual(mid, 0.6, places=6)          # reference by construction
        self.assertLess(high, mid)
        self.assertLess(mid, low)
        self.assertTrue(0.0 <= high < 1.0 and 0.0 < low < 1.0)
        ep = effective_params("satellite-ground", {"noise_level": 0.005, "elevation": 45.0})
        self.assertAlmostEqual(ep["ad_rate"], 0.6, places=6)
        self.assertAlmostEqual(ep["pd_rate"], 0.05, places=6)
        self.assertAlmostEqual(ep["n_th"], 0.1, places=6)

    def test_noise_level_scales_channels(self):
        base = effective_params("deep-space", {"noise_level": 0.05})["p2"]
        double = effective_params("deep-space", {"noise_level": 0.10})["p2"]
        self.assertAlmostEqual(base, 0.05, places=6)
        self.assertAlmostEqual(double, 0.10, places=6)      # linear in level/REP

    def test_custom_uses_absolute_params(self):
        ep = effective_params("custom", {"p1": 0.01, "p2": 0.02, "readout": 0.03,
                                         "burst_rate": 0.04})
        self.assertAlmostEqual(ep["p1"], 0.01, places=9)
        self.assertAlmostEqual(ep["p2"], 0.02, places=9)
        self.assertAlmostEqual(ep["readout"], 0.03, places=9)
        self.assertAlmostEqual(ep["burst_rate"], 0.04, places=9)

    def test_environment_info_metadata(self):
        info = get_environment_info("leo-satellite")
        for key in ("name", "description", "key_params", "citation", "missions"):
            self.assertIn(key, info)
        self.assertTrue(info["citation"])
        self.assertTrue(len(info["missions"]) >= 1)
        allinfo = get_environment_info()
        self.assertEqual(len(allinfo), len(ENVIRONMENTS))
        with self.assertRaises(ValueError):
            get_environment_info("not-an-env")

    def test_earth_lab_backward_compatible(self):
        # Phase 1 behaviour preserved: p1 = p2 = readout = noise_level.
        for lvl in (0.01, 0.03, 0.06):
            s = noise_summary("earth-lab", {"noise_level": lvl})
            self.assertAlmostEqual(s["p1"], lvl, places=9)
            self.assertAlmostEqual(s["p2"], lvl, places=9)
            self.assertAlmostEqual(s["readout"], lvl, places=9)

    def test_space_models_run_on_aer(self):
        # Each space model must actually execute a density-matrix simulation and
        # yield a normalised distribution (valid CPTP channels).
        G = generate_graph("3-regular", 4, seed=3)
        for env in SPACE_ENVIRONMENTS:
            nm = get_noise_model(env, {"noise_level": REPRESENTATIVE_LEVELS[env]})
            r = evaluate_qaoa_exact(G, [0.4], [0.3], noise_model=nm, shots=512)
            self.assertAlmostEqual(sum(r["counts"].values()), 512, delta=2)
            self.assertGreaterEqual(r["approximation_ratio"], 0.0)
            self.assertLessEqual(r["approximation_ratio"], 1.0 + 1e-9)


class TestResonanceDetector(unittest.TestCase):
    """The pure model-comparison logic behind the Noise Analysis page."""

    def test_monotone_decay_no_resonance(self):
        from utils.analysis import detect_resonance
        x = np.array([0.0, 0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1])
        y = 0.84 * np.exp(-x / 0.06) + 0.02
        r = detect_resonance(x, y)
        self.assertFalse(r["resonance"])
        self.assertEqual(r["winner"], "monotone")
        self.assertEqual(r["p_star"], 0.0)

    def test_interior_peak_detected(self):
        from utils.analysis import detect_resonance
        x = np.array([0.0, 0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1])
        y = 0.80 + 0.06 * np.exp(-((x - 0.02) ** 2) / (2 * 0.02 ** 2)) - 0.15 * x
        r = detect_resonance(x, y)
        self.assertTrue(r["resonance"])
        self.assertEqual(r["winner"], "peak")
        self.assertTrue(0.0 < r["p_star"] < 0.1)
        self.assertGreater(r["improvement_pct"], 0.0)

    def test_degradation_stats_and_slice(self):
        from utils.analysis import degradation_stats, slice_at_gamma
        x = np.array([0.0, 0.05, 0.1])
        y = np.array([0.80, 0.70, 0.60])
        d = degradation_stats(x, y)
        self.assertAlmostEqual(d["pct_loss_max"], 25.0, places=6)   # (0.8-0.6)/0.8
        g = np.linspace(0, 2, 5)
        b = np.linspace(0, 1, 4)
        GG, BB = np.meshgrid(g, b, indexing="ij")
        L = {"gammas": g, "betas": b, "ratios": GG * 0 + BB}
        betas, row = slice_at_gamma(L, 1.0)
        self.assertEqual(len(row), len(b))
        self.assertTrue(np.allclose(row, b))                        # ratio = beta, gamma-independent


class TestClassicalBaselines(unittest.TestCase):
    """The classical MaxCut baselines used by the Advantage Map (pure, no Qiskit)."""

    def test_gw_is_constant_guarantee(self):
        G = generate_graph("3-regular", 6, seed=42)
        self.assertAlmostEqual(goemans_willison_ratio(G), GW_GUARANTEE, places=6)
        self.assertAlmostEqual(GW_GUARANTEE, 0.8746, places=6)

    def test_annealing_finds_optimum_on_small_graph(self):
        G = generate_graph("3-regular", 6, seed=42)          # Cmax = 7
        r = simulated_annealing_maxcut(G, n_runs=20, seed=0)
        self.assertGreaterEqual(r, 0.999)                    # solves it optimally

    def test_random_matches_half_edges_over_cmax(self):
        G = generate_graph("3-regular", 8, seed=3)
        cmax, _ = optimal_maxcut(G)
        expected = (G.number_of_edges() / 2.0) / cmax
        r = random_sampling_ratio(G, n_samples=20000, seed=1)
        self.assertAlmostEqual(r, expected, delta=0.02)      # MC close to |E|/(2 Cmax)

    def test_all_ratios_in_unit_interval(self):
        G = generate_graph("erdos-renyi", 10, seed=5, p=0.5)
        b = all_baselines(G, sa_runs=10, samples=2000, seed=2)
        for k, v in b.items():
            self.assertGreaterEqual(v, 0.0, k)
            self.assertLessEqual(v, 1.0 + 1e-9, k)
        self.assertAlmostEqual(b["best"], max(b["GW"], b["annealing"], b["random"]), places=9)

    def test_best_classical_selector_and_aliases(self):
        G = generate_graph("3-regular", 6, seed=42)
        self.assertAlmostEqual(best_classical_ratio(G, method="GW"), GW_GUARANTEE, places=6)
        self.assertAlmostEqual(best_classical_ratio(G, method="SA"),
                               best_classical_ratio(G, method="annealing"), places=9)
        self.assertGreaterEqual(best_classical_ratio(G, method="best"),
                                best_classical_ratio(G, method="GW"))

    def test_unknown_method_raises(self):
        G = generate_graph("3-regular", 6, seed=42)
        with self.assertRaises(ValueError):
            best_classical_ratio(G, method="not-a-method")

    def test_seeded_determinism(self):
        G = generate_graph("3-regular", 8, seed=11)
        self.assertEqual(simulated_annealing_maxcut(G, seed=7),
                         simulated_annealing_maxcut(G, seed=7))
        self.assertEqual(random_sampling_ratio(G, seed=7),
                         random_sampling_ratio(G, seed=7))

    def test_degenerate_graph_no_edges(self):
        import networkx as nx
        G = nx.Graph()
        G.add_nodes_from(range(4))                            # 4 isolated vertices
        self.assertEqual(random_sampling_ratio(G), 0.0)
        self.assertEqual(simulated_annealing_maxcut(G), 0.0)


class TestAdvantageMap(unittest.TestCase):
    """The quantum-advantage boundary computation (fixed-angle protocol)."""

    def test_valid_sizes_even_for_three_regular_and_capped(self):
        self.assertEqual(valid_sizes("3-regular", 4, 16), [4, 6, 8, 10, 12])
        self.assertEqual(valid_sizes("erdos-renyi", 4, 7), [4, 5, 6, 7])
        self.assertTrue(max(valid_sizes("3-regular", 4, 99)) <= MEMORY_MAX_N)

    def test_estimate_cost_grows_with_size_depth_envs(self):
        small = estimate_map_cost([4, 6], 1, ["earth-lab"], [0.01, 0.05], 2)
        big = estimate_map_cost([4, 6, 8, 10], 3, MAP_ENVIRONMENTS, [0.01, 0.05], 3)
        self.assertGreater(big, small)
        self.assertGreater(small, 0.0)

    def test_reference_angles_return_correct_length_and_ratio(self):
        G = generate_graph("3-regular", 6, seed=42)
        for depth in (1, 2):
            gammas, betas, ratio = reference_angles(G, depth=depth, restarts=2, max_iter=40)
            self.assertEqual(len(gammas), depth)
            self.assertEqual(len(betas), depth)
            self.assertGreater(ratio, 0.5)                    # beats random on n=6

    def test_compute_map_shape_and_physics(self):
        sizes = [4, 6]
        noise = [0.001, 0.01, 0.05, 0.1]
        envs = ["earth-lab", "leo-satellite"]
        res = compute_advantage_map(
            "3-regular", sizes, depth=2, envs=envs, baseline="GW",
            noise_levels=noise, n_instances=1, seed_base=42,
            opt_restarts=2, opt_max_iter=40, shots=128,
        )
        self.assertEqual(res["sizes"], sizes)
        self.assertEqual(res["envs"], envs)
        # difference grid is (n_sizes, n_noise) per env
        for e in envs:
            D = np.asarray(res["difference"][e])
            self.assertEqual(D.shape, (len(sizes), len(noise)))
            self.assertIn(str(sizes[0]), res["boundary"][e])
        # QAOA ratio must fall as noise rises (fixed angles, depolarising env)
        q4 = res["qaoa"]["earth-lab"]["4"]
        self.assertGreater(q4[0], q4[-1])
        # n=4 at depth 2 beats GW at the lowest noise -> advantage somewhere
        self.assertTrue(res["advantage"]["earth-lab"]["any_advantage"])

    def test_json_serialisable(self):
        import json
        res = compute_advantage_map(
            "3-regular", [4], depth=1, envs=["earth-lab"], baseline="GW",
            noise_levels=[0.001, 0.05], n_instances=1, seed_base=1,
            opt_restarts=1, opt_max_iter=25, shots=64,
        )
        s = json.dumps(res)                                   # must not raise
        self.assertGreater(len(s), 50)

    def test_summarize_no_advantage(self):
        sizes = [4, 6]
        noise = [0.001, 0.1]
        D = np.array([[-0.1, -0.2], [-0.3, -0.4]])            # classical always ahead
        bnd = {"4": None, "6": None}
        s = summarize_advantage(sizes, noise, D, bnd, "earth-lab", "GW")
        self.assertFalse(s["any_advantage"])

    def test_summarize_full_range_advantage(self):
        sizes = [4, 6]
        noise = [0.001, 0.1]
        D = np.array([[0.2, 0.1], [0.15, 0.05]])              # quantum ahead everywhere
        bnd = {"4": float("inf"), "6": float("inf")}
        s = summarize_advantage(sizes, noise, D, bnd, "earth-lab", "GW")
        self.assertTrue(s["any_advantage"])
        self.assertEqual(s["n_range"], [4, 6])

    def test_unknown_environment_raises(self):
        with self.assertRaises(ValueError):
            compute_advantage_map("3-regular", [4], depth=1, envs=["not-an-env"],
                                  baseline="GW", noise_levels=[0.01], n_instances=1)


class TestBarrenPlateau(unittest.TestCase):
    """Phase 4: landscape-shape trainability and the Barren Plateau diagram."""

    def test_trainability_score_flat_vs_structured(self):
        flat = np.full((7, 7), 0.62)
        structured = np.array([
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ])
        self.assertEqual(compute_trainability_score(flat), 0.0)
        self.assertGreater(compute_trainability_score(structured), 0.9)
        m = landscape_metrics(flat)
        self.assertEqual(m["variance"], 0.0)
        self.assertEqual(m["max_gradient"], 0.0)
        self.assertEqual(m["ratio_range"], 0.0)

    def test_classification_thresholds_and_resonant_candidate(self):
        self.assertEqual(classify_trainability(0.61, 0.0, 0.0), "TRAINABLE")
        self.assertEqual(classify_trainability(0.09, 0.0, 0.0), "BARREN")
        self.assertEqual(classify_trainability(0.20, 0.001, 0.20), "MIXED")
        # Intermediate score + meaningful variance and slope = a structured
        # noise-adjusted resonance candidate (the diagram later verifies noise evidence).
        self.assertEqual(classify_trainability(0.40, 0.010, 1.00), "RESONANT")

    def test_axes_and_cost_grow_with_problem(self):
        self.assertEqual(diagram_sizes("3-regular", (4, 16)), [4, 6, 8, 10, 12, 14, 16])
        self.assertEqual(diagram_depths((1, 8)), list(range(1, 9)))
        with self.assertRaises(ValueError):
            diagram_sizes("3-regular", [5, 7])
        spec = {"environment": "earth-lab", "params": {"noise_level": 0.003}}
        small = estimate_barren_cost("3-regular", [4, 6], [1, 2], spec,
                                     n_instances=1, resolution=9)
        big = estimate_barren_cost("3-regular", (4, 16), (1, 8), spec,
                                   n_instances=3, resolution=10)
        self.assertGreater(big["seconds"], small["seconds"])
        self.assertGreater(big["transfer_jobs"], 0)
        self.assertGreater(big["exact_noise_jobs"], 0)

    def test_compute_diagram_shape_progress_and_json(self):
        import json
        calls = []
        spec = {"environment": "earth-lab", "params": {"noise_level": 0.003}}
        result = compute_barren_diagram(
            "3-regular", [4, 6], [1, 2], spec, n_instances=1,
            seed_base=7, resolution=9, shots=64,
            progress_callback=lambda done, total: calls.append((done, total)),
        )
        self.assertEqual(result["sizes"], [4, 6])
        self.assertEqual(result["depths"], [1, 2])
        self.assertEqual(np.asarray(result["scores"]).shape, (2, 2))
        self.assertEqual(np.asarray(result["metrics"]["variance"]).shape, (2, 2))
        self.assertEqual(result["scores"], result["trainability_scores"])
        self.assertEqual(result["classifications"], result["classification_grid"])
        self.assertTrue(all(c in {"BARREN", "MIXED", "TRAINABLE", "RESONANT"}
                            for row in result["classifications"] for c in row))
        self.assertTrue(all(mode == "exact-noisy" for row in result["source"]["mode"] for mode in row))
        self.assertTrue(calls)
        self.assertAlmostEqual(calls[-1][0], calls[-1][1])
        self.assertGreater(len(json.dumps(result)), 100)

    def test_earth_low_noise_trainable_leo_shrinks_and_transfer_is_explicit(self):
        common = dict(
            family="3-regular", n_range=[4, 10], p_range=[1, 4, 8],
            n_instances=1, seed_base=42, resolution=9, shots=64,
        )
        earth = compute_barren_diagram(
            noise_model={"environment": "earth-lab", "params": {"noise_level": 0.003}}, **common
        )
        leo = compute_barren_diagram(
            noise_model={"environment": "leo-satellite", "params": {"noise_level": 0.008}}, **common
        )
        earth_scores = np.asarray(earth["scores"])
        leo_scores = np.asarray(leo["scores"])
        # Verification 1: shallow / small-n Earth terrain is trainable while a
        # large, deep cell is barren after concentration.
        self.assertGreater(earth_scores[0, 0], 0.6)
        self.assertLess(earth_scores[-1, -1], 0.1)
        # Verification 2: LEO's stronger channels reduce usable terrain.
        self.assertLess(float(leo_scores.mean()), float(earth_scores.mean()))
        compared = compare_diagrams({"earth-lab": earth, "leo-satellite": leo})
        self.assertGreater(compared["comparison"]["shrink_percent"], 0.0)
        # n=10 deliberately avoids an unsafe density-matrix allocation.
        self.assertTrue(all(mode == "contrast-transfer" for row in earth["source"]["mode"]
                            for mode in [row[1]]))

    def test_exact_noisy_memory_guard(self):
        spec = {"environment": "earth-lab", "params": {"noise_level": 0.003}}
        with self.assertRaises(ValueError):
            compute_barren_diagram("3-regular", [14], [1], spec, n_instances=1,
                                   execution="exact", resolution=9)
        self.assertEqual(EXACT_NOISE_MEMORY_MAX_N, 12)
        self.assertEqual(DEFAULT_EXACT_NOISE_MAX_N, 8)


class TestHardwareProfiles(unittest.TestCase):
    """Phase 5: local IBM Runtime FakeProvider calibration integration."""

    def test_available_profiles_and_cairo_stats(self):
        self.assertEqual(set(AVAILABLE_HARDWARE),
                         {"ibm_cairo", "ibm_kyoto", "ibm_sherbrooke", "ibm_hanoi"})
        stats = extract_hardware_stats("ibm_cairo", subset_size=4)
        for key in ("n_qubits", "avg_cx_error", "avg_single_qubit_error", "avg_T1", "avg_T2", "readout_error"):
            self.assertIn(key, stats)
        self.assertEqual(stats["n_qubits"], 27)
        self.assertEqual(stats["subset_size"], 4)
        self.assertTrue(0.0 < stats["avg_cx_error"] < 0.5)
        self.assertTrue(0.0 < stats["avg_single_qubit_error"] < 0.1)
        self.assertTrue(stats["avg_T1"] > 0.0 and stats["avg_T2"] > 0.0)
        self.assertTrue(0.0 <= stats["readout_error"] < 0.5)

    def test_full_and_qaoa_native_reduced_noise_models(self):
        from qiskit_aer.noise import NoiseModel
        full = get_hardware_noise_model("ibm_cairo")
        reduced = get_hardware_noise_model("ibm_cairo", subset_size=4)
        self.assertIsInstance(full, NoiseModel)
        self.assertIsInstance(reduced, NoiseModel)
        self.assertEqual(full._qc_hardware_mode, "full-fake-backend")
        self.assertEqual(reduced._qc_hardware_mode, "reduced-qaoa-native")
        self.assertTrue({"h", "rx", "rzz", "measure"}.issubset(set(reduced.noise_instructions)))
        self.assertEqual(hardware_noise_spec("ibm_cairo", 4),
                         {"hardware_backend": "ibm_cairo", "subset_size": 4})

    def test_cairo_hardware_descriptor_degrades_real_landscape(self):
        G = generate_graph("3-regular", 4, seed=42)
        ideal = compute_landscape(G, depth=1, resolution=9, noise_model=None, shots=64)
        hardware = compute_landscape(
            G, depth=1, resolution=9, noise_model=hardware_noise_spec("ibm_cairo", 4), shots=64
        )
        self.assertLess(hardware["best_ratio"], ideal["best_ratio"])
        self.assertLess(np.ptp(hardware["ratios"]), np.ptp(ideal["ratios"]))


class TestMissionPlanner(unittest.TestCase):
    """Phase 5: pure, exportable space-QAOA mission recommendations."""

    def test_templates_are_complete_and_independent(self):
        names = ["spoqc_like", "iss_quantum", "lunar_gateway", "mars_orbiter"]
        for name in names:
            template = get_mission_template(name)
            for key in ("orbit", "qubit_tech", "T1", "T2", "gate_fidelity", "available_qubits", "duration", "shielding"):
                self.assertIn(key, template)
        first = get_mission_template("spoqc_like")
        first["T1"] = -1
        self.assertGreater(get_mission_template("spoqc_like")["T1"], 0.0)

    def test_spoqc_plan_has_requested_sections_and_json(self):
        import json
        plan = generate_mission_plan(get_mission_template("spoqc_like"))
        for key in ("environment_assessment", "recommended_strategy", "risk_assessment", "advantage_window"):
            self.assertIn(key, plan)
        env = plan["environment_assessment"]
        self.assertGreater(env["estimated_effective_noise"], 0.0)
        self.assertGreater(env["radiation_events_per_day"], 0.0)
        self.assertGreaterEqual(plan["recommended_strategy"]["optimal_depth"], 1)
        self.assertEqual(plan["advantage_window"]["n_range"][0], 4)
        self.assertGreater(len(json.dumps(plan)), 500)

    def test_mars_orbiter_is_more_conservative_than_spoqc(self):
        spoqc = generate_mission_plan(get_mission_template("spoqc_like"))
        mars = generate_mission_plan(get_mission_template("mars_orbiter"))
        self.assertGreater(mars["environment_assessment"]["radiation_events_per_day"],
                           spoqc["environment_assessment"]["radiation_events_per_day"])
        self.assertLessEqual(mars["recommended_strategy"]["optimal_depth"],
                             spoqc["recommended_strategy"]["optimal_depth"])
        self.assertLessEqual(mars["advantage_window"]["n_range"][1],
                             spoqc["advantage_window"]["n_range"][1])

    def test_invalid_inputs_and_pdf_export(self):
        bad = get_mission_template("spoqc_like")
        bad["gate_fidelity"] = 1.2
        with self.assertRaises(ValueError):
            generate_mission_plan(bad)
        from utils.export import export_analysis_pdf
        payload = export_analysis_pdf(
            "Mission Plan Test", "PDF export verification",
            [{"type": "text", "heading": "Summary", "lines": ["SPOQC-like plan generated."]}],
        )
        self.assertTrue(payload.startswith(b"%PDF"))
        self.assertGreater(len(payload), 500)


class TestExperimentalValidation(unittest.TestCase):
    """Research-validation protocol: immutable manifests, data integrity, and dry-run circuits."""

    @staticmethod
    def _manifest():
        import networkx as nx

        return create_experiment_manifest(
            nx.cycle_graph(4), graph_metadata={"family": "cycle"},
            gamma_values=[0.0, 0.5], beta_values=[0.0, 0.25], shots=1024,
            randomization_seed=19, bootstrap_resamples=250,
            created_utc="2026-10-26T00:00:00+00:00",
        )

    @staticmethod
    def _counts_csv(manifest):
        rows = ["manifest_id,condition_id,bitstring,count"]
        for condition in manifest["execution_plan"]["conditions"]:
            rows.extend([
                f"{manifest['manifest_id']},{condition['condition_id']},0000,512",
                f"{manifest['manifest_id']},{condition['condition_id']},0101,512",
            ])
        return ("\n".join(rows) + "\n").encode("utf-8")

    def test_manifest_round_trip_checksum_and_count_conventions(self):
        import json

        manifest = self._manifest()
        restored = parse_experiment_manifest(manifest_to_json(manifest))
        self.assertEqual(restored["manifest_id"], manifest["manifest_id"])
        self.assertEqual(normalize_bitstring("0101", 4, "qiskit-classical-order"), "1010")
        frame = parse_counts_csv(self._counts_csv(manifest), manifest)
        self.assertTrue((frame["bitstring"] == "0000").any())  # leading zeroes survive CSV parsing
        self.assertEqual(frame["count"].sum(), 1024 * len(manifest["execution_plan"]["conditions"]))

        tampered = json.loads(manifest_to_json(manifest))
        tampered["execution_plan"]["shots_per_condition"] = 2048
        with self.assertRaises(ExperimentalValidationError):
            parse_experiment_manifest(tampered)
        with self.assertRaises(ExperimentalValidationError):
            parse_counts_csv(b"condition_id,bitstring,count\nunknown,0000,4\n", manifest)

    def test_bootstrap_distribution_metrics_and_analysis_report(self):
        manifest = self._manifest()
        frame = parse_counts_csv(self._counts_csv(manifest), manifest)
        observed = {
            condition_id: dict(zip(group["bitstring"], group["count"]))
            for condition_id, group in frame.groupby("condition_id")
        }
        analysis = analyze_validation_study(manifest, frame, {"identical_control": observed}, bootstrap_seed=5)
        summary = analysis["model_summary"]["identical_control"]
        self.assertEqual(summary["weighted_mean_absolute_ratio_error"], 0.0)
        self.assertEqual(summary["weighted_mean_total_variation_distance"], 0.0)
        self.assertEqual(distribution_metrics({"0000": 10}, {"1111": 10}, 4)["total_variation_distance"], 1.0)

        import networkx as nx
        interval = bootstrap_ratio_interval({"0000": 512, "0101": 512}, nx.cycle_graph(4), resamples=250, seed=3)
        self.assertLessEqual(interval["lower"], interval["approximation_ratio"])
        self.assertGreaterEqual(interval["upper"], interval["approximation_ratio"])
        report = validation_report_markdown(manifest, analysis)
        self.assertIn("Experimental Validation Report", report)
        self.assertIn("does not identify a physical noise source", report)

    def test_qiskit_raw_count_output_is_normalized_for_analysis(self):
        from scripts.submit_ibm_validation import write_run_outputs

        manifest = self._manifest()
        conditions = sorted(manifest["execution_plan"]["conditions"], key=lambda item: item["run_order"])
        # Raw Qiskit c[n-1]...c[0] key 0001 becomes vertex-order 1000.
        raw_counts = [{"0001": 1024} for _ in conditions]
        with tempfile.TemporaryDirectory() as directory:
            counts_path, raw_path, metadata_path = write_run_outputs(
                Path(directory), manifest, conditions, raw_counts,
                backend_name="unit-test-backend", job_id="unit-test-job", qiskit_version="test",
                optimization_level=1, overwrite=False,
            )
            self.assertTrue(raw_path.exists() and metadata_path.exists())
            frame = parse_counts_csv(counts_path.read_bytes(), manifest)
            self.assertEqual(set(frame["bitstring"]), {"1000"})
            self.assertEqual(frame["count"].sum(), 1024 * len(conditions))

    def test_runtime_submission_cli_dry_run_has_no_credential_requirement(self):
        import json

        manifest = self._manifest()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(manifest_to_json(manifest), encoding="utf-8")
            output_dir = root / "dry_run"
            completed = subprocess.run(
                [sys.executable, "scripts/submit_ibm_validation.py", "--manifest", str(manifest_path),
                 "--output-dir", str(output_dir), "--dry-run"],
                cwd=_ROOT, text=True, capture_output=True, timeout=90,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            plan = json.loads((output_dir / "dry_run_submission_plan.json").read_text(encoding="utf-8"))
            self.assertFalse(plan["network_submission"])
            self.assertEqual(len(plan["circuits"]), manifest["execution_plan"]["condition_count"])
            self.assertTrue(all("measure" in item["operation_counts"] for item in plan["circuits"]))


class TestLandscapeFingerprinting(unittest.TestCase):
    """Phase 6: finite shape features, RF diagnostics, and cached inference."""

    def test_feature_vector_is_complete_finite_and_degenerate_safe(self):
        surface = np.arange(81, dtype=float).reshape(9, 9)
        fingerprint = extract_landscape_fingerprint(surface)
        self.assertEqual(fingerprint.shape, (len(FINGERPRINT_FEATURE_NAMES),))
        self.assertTrue(np.isfinite(fingerprint).all())
        self.assertTrue(np.isfinite(extract_landscape_fingerprint(np.ones((3, 3)))).all())
        with self.assertRaises(ValueError):
            extract_landscape_fingerprint(np.array([1.0, 2.0]))

    def test_reduced_random_forest_training_and_prediction_contract(self):
        from sklearn.ensemble import RandomForestClassifier

        graph = generate_graph("3-regular", 4, seed=42)
        model = train_fingerprint_classifier(
            graph, ["ideal", "earth-lab"], n_per_env=2, resolution=3, seed=7
        )
        self.assertIsInstance(model, RandomForestClassifier)
        self.assertEqual(set(model.classes_), {"ideal", "earth-lab"})
        landscape = compute_landscape(
            graph, depth=1, resolution=3,
            noise_model={"environment": "earth-lab", "params": {"noise_level": 0.003}},
            shots=64,
        )
        prediction = predict_environment(landscape["ratios"], model)
        for key in ("predicted_environment", "confidence", "top_3", "top_three",
                    "estimated_params", "estimated_parameters", "features"):
            self.assertIn(key, prediction)
        self.assertGreaterEqual(prediction["confidence"], 0.0)
        self.assertLessEqual(prediction["confidence"], 100.0)
        self.assertLessEqual(len(prediction["top_three"]), 2)
        self.assertIn("p2", prediction["estimated_parameters"])
        self.assertIn("burst_rate", prediction["estimated_parameters"])

    def test_default_cache_and_mystery_catalog(self):
        self.assertEqual(len(DEFAULT_FINGERPRINT_ENVIRONMENTS), 6)
        catalog = mystery_landscape_catalog()
        self.assertEqual(len(catalog), 5)
        self.assertTrue(all({"id", "title", "hint"}.issubset(item) for item in catalog))
        model = load_default_classifier()
        self.assertEqual(set(model.classes_), set(DEFAULT_FINGERPRINT_ENVIRONMENTS))
        payload = generate_mystery_landscape(catalog[0]["id"], resolution=3)
        self.assertEqual(payload["landscape"]["ratios"].shape, (3, 3))
        self.assertTrue(np.isfinite(payload["landscape"]["ratios"]).all())


class TestStaticExports(unittest.TestCase):
    """Phase 6: static Plotly fallback and real PDF/CSV file deliverables."""

    def test_static_plotly_exports_csv_and_report_pdf(self):
        import plotly.graph_objects as go
        from utils.export import (
            export_data_csv,
            export_figure_pdf,
            export_figure_png,
            export_figure_svg,
            export_report_pdf,
        )

        gammas = np.linspace(0.0, 1.0, 3)
        betas = np.linspace(0.0, 1.0, 3)
        ratios = np.outer(gammas + 0.2, betas + 0.3)
        figure = go.Figure(go.Heatmap(x=betas, y=gammas, z=ratios))
        figure.update_layout(title="Export regression test")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            png = Path(export_figure_png(figure, str(root / "landscape.png")))
            svg = Path(export_figure_svg(figure, str(root / "landscape.svg")))
            pdf = Path(export_figure_pdf(figure, str(root / "landscape.pdf")))
            csv = Path(export_data_csv(gammas, betas, ratios, str(root / "landscape.csv")))
            report = Path(export_report_pdf(
                {"resolution": 3}, {"best_ratio": float(ratios.max())}, [figure],
                str(root / "report.pdf"),
            ))
            self.assertTrue(png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
            self.assertIn("<svg", svg.read_text(encoding="utf-8")[:1000])
            self.assertTrue(pdf.read_bytes().startswith(b"%PDF"))
            self.assertTrue(report.read_bytes().startswith(b"%PDF"))
            self.assertEqual(csv.read_text(encoding="utf-8").splitlines()[0], "gamma,beta,ratio")


if __name__ == "__main__":
    unittest.main(verbosity=2)
