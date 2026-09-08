"""
Evaluation metrics and benchmarking suite.
"""

from lunar_registration.evaluation.metrics import (
    calculate_rmse,
    calculate_inlier_ratio,
    calculate_mae,
    compute_spatial_coverage_metric,
    compute_per_cell_inliers,
)
from lunar_registration.evaluation.visualization import draw_matches
from lunar_registration.evaluation.benchmark import (
    BenchmarkSuite,
    BenchmarkResult,
    run_standard_benchmark,
)

__all__ = [
    "calculate_rmse",
    "calculate_inlier_ratio",
    "calculate_mae",
    "compute_spatial_coverage_metric",
    "compute_per_cell_inliers",
    "draw_matches",
    "BenchmarkSuite",
    "BenchmarkResult",
    "run_standard_benchmark",
]
