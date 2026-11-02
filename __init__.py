"""
Spatiotemporal Mobility Trajectory Privacy Auditing Package.
Federated Learning with NextLocationModel, H3 Discretization, and Tab Attack.
"""

from .model import NextLocationModel
from .dataset import (
    CANARY_HEX_SEQUENCE,
    CANARY_STRING,
    LocationTokenizer,
    get_client_dataloaders,
    generate_synthetic_gps_data,
    discretize_gps_trajectories,
)
from .attack import (
    compute_sequence_perplexity,
    greedy_rollout_trajectory,
    tab_attack_leakage_analysis,
    get_quantitative_privacy_metrics,
    print_quantitative_report,
)

__all__ = [
    "NextLocationModel",
    "CANARY_HEX_SEQUENCE",
    "CANARY_STRING",
    "LocationTokenizer",
    "get_client_dataloaders",
    "generate_synthetic_gps_data",
    "discretize_gps_trajectories",
    "compute_sequence_perplexity",
    "greedy_rollout_trajectory",
    "tab_attack_leakage_analysis",
    "get_quantitative_privacy_metrics",
    "print_quantitative_report",
]

