"""
Utility modules: logging and visualization.
"""

from lunar_registration.utils.logging import setup_logging, get_logger
from lunar_registration.utils.visualization import (
    plot_side_by_side,
    plot_matches_overlay,
    plot_difference_heatmap,
    plot_registration_summary,
)

__all__ = [
    "setup_logging",
    "get_logger",
    "plot_side_by_side",
    "plot_matches_overlay",
    "plot_difference_heatmap",
    "plot_registration_summary",
]
