"""Quantum Compass — core QAOA engine.

This package contains the physics and numerics behind the Streamlit app:

* :mod:`core.graph_generator`  – problem-instance (graph) construction + MaxCut ground truth
* :mod:`core.qaoa_builder`     – QAOA ansatz circuit construction
* :mod:`core.noise_models`     – environment noise models (ideal / earth-lab / 5 space envs)
* :mod:`core.qaoa_evaluator`   – single-point QAOA evaluation on Aer
* :mod:`core.landscape`        – 2-D (gamma, beta) approximation-ratio landscapes
* :mod:`core.optimizer`        – multi-start classical optimizers over the angles
* :mod:`core.classical_baselines` – pure MaxCut reference solvers / guarantees
* :mod:`core.advantage_map`    – QAOA-vs-classical advantage boundaries
* :mod:`core.barren_plateau`   – landscape-trainability phase diagrams
* :mod:`core.hardware_profiles` – IBM fake-backend calibration profiles
* :mod:`core.mission_planner`  – pure space-mission operating recommendations
* :mod:`core.fingerprint`      – landscape-shape environment diagnostics
* :mod:`core.experimental_validation` – reproducible hardware/simulator validation protocol

The public helpers are re-exported here for convenience.
"""

from core.graph_generator import (
    generate_graph,
    optimal_maxcut,
    compute_graph_properties,
    cut_value,
    GRAPH_FAMILIES,
)
from core.qaoa_builder import build_qaoa_circuit
from core.noise_models import (
    get_noise_model,
    get_environment_info,
    effective_params,
    noise_summary,
    eta_from_elevation,
    ENVIRONMENTS,
    SPACE_ENVIRONMENTS,
    REPRESENTATIVE_LEVELS,
)
from core.qaoa_evaluator import (
    evaluate_qaoa,
    evaluate_qaoa_exact,
    cost_operator,
)
from core.landscape import (
    compute_landscape,
    detect_local_minima,
    detect_local_maxima,
)
from core.optimizer import run_optimization
from core.classical_baselines import (
    goemans_willison_ratio,
    simulated_annealing_maxcut,
    random_sampling_ratio,
    best_classical_ratio,
    all_baselines,
    GW_GUARANTEE,
)
from core.advantage_map import (
    compute_advantage_map,
    reference_angles,
    estimate_map_cost,
    valid_sizes,
    MAP_ENVIRONMENTS,
    MEMORY_MAX_N,
)
from core.barren_plateau import (
    compute_trainability_score,
    classify_trainability,
    landscape_metrics,
    compute_barren_diagram,
    estimate_barren_cost,
    diagram_sizes,
    diagram_depths,
    compare_diagrams,
    TRAINABILITY_CLASSES,
    EXACT_NOISE_MEMORY_MAX_N,
    DEFAULT_EXACT_NOISE_MAX_N,
)
from core.hardware_profiles import (
    AVAILABLE_HARDWARE,
    get_hardware_noise_model,
    extract_hardware_stats,
    hardware_noise_spec,
)
from core.mission_planner import (
    generate_mission_plan,
    get_mission_template,
)
from core.fingerprint import (
    FINGERPRINT_FEATURE_NAMES,
    DEFAULT_FINGERPRINT_ENVIRONMENTS,
    extract_landscape_fingerprint,
    fingerprint_feature_dict,
    train_fingerprint_classifier,
    load_default_classifier,
    predict_environment,
    mystery_landscape_catalog,
    generate_mystery_landscape,
)
from core.experimental_validation import (
    MANIFEST_KIND,
    MANIFEST_SCHEMA_VERSION,
    ExperimentalValidationError,
    standard_angle_grid,
    create_experiment_manifest,
    validate_experiment_manifest,
    manifest_to_json,
    parse_experiment_manifest,
    parse_counts_csv,
    bootstrap_ratio_interval,
    distribution_metrics,
    analyze_validation_study,
    validation_report_markdown,
)

__all__ = [
    "generate_graph",
    "optimal_maxcut",
    "compute_graph_properties",
    "GRAPH_FAMILIES",
    "build_qaoa_circuit",
    "get_noise_model",
    "get_environment_info",
    "effective_params",
    "noise_summary",
    "eta_from_elevation",
    "ENVIRONMENTS",
    "SPACE_ENVIRONMENTS",
    "REPRESENTATIVE_LEVELS",
    "evaluate_qaoa",
    "evaluate_qaoa_exact",
    "cut_value",
    "cost_operator",
    "compute_landscape",
    "detect_local_minima",
    "detect_local_maxima",
    "run_optimization",
    "goemans_willison_ratio",
    "simulated_annealing_maxcut",
    "random_sampling_ratio",
    "best_classical_ratio",
    "all_baselines",
    "GW_GUARANTEE",
    "compute_advantage_map",
    "reference_angles",
    "estimate_map_cost",
    "valid_sizes",
    "MAP_ENVIRONMENTS",
    "MEMORY_MAX_N",
    "compute_trainability_score",
    "classify_trainability",
    "landscape_metrics",
    "compute_barren_diagram",
    "estimate_barren_cost",
    "diagram_sizes",
    "diagram_depths",
    "compare_diagrams",
    "TRAINABILITY_CLASSES",
    "EXACT_NOISE_MEMORY_MAX_N",
    "DEFAULT_EXACT_NOISE_MAX_N",
    "AVAILABLE_HARDWARE",
    "get_hardware_noise_model",
    "extract_hardware_stats",
    "hardware_noise_spec",
    "generate_mission_plan",
    "get_mission_template",
    "FINGERPRINT_FEATURE_NAMES",
    "DEFAULT_FINGERPRINT_ENVIRONMENTS",
    "extract_landscape_fingerprint",
    "fingerprint_feature_dict",
    "train_fingerprint_classifier",
    "load_default_classifier",
    "predict_environment",
    "mystery_landscape_catalog",
    "generate_mystery_landscape",
    "MANIFEST_KIND",
    "MANIFEST_SCHEMA_VERSION",
    "ExperimentalValidationError",
    "standard_angle_grid",
    "create_experiment_manifest",
    "validate_experiment_manifest",
    "manifest_to_json",
    "parse_experiment_manifest",
    "parse_counts_csv",
    "bootstrap_ratio_interval",
    "distribution_metrics",
    "analyze_validation_study",
    "validation_report_markdown",
]
