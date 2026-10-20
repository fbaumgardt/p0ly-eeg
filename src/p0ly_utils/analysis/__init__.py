"""Analysis modules for p0ly_utils (US-010+): ERP, then PSD/TFR/connectivity/...

Each analysis domain lives in its own submodule. All functions are pure
(stateless, no filesystem paths) and documented with array shapes per
[[SCHEMA]] §1.
"""

from p0ly_utils.analysis.erp import (
    ClusterPermStats,
    GroupERPResult,
    average_epochs,
    cluster_permutation_1samp,
    eeg_adjacency,
    evoked_difference,
    grand_average,
    group_erp,
)

__all__ = [
    "ClusterPermStats",
    "GroupERPResult",
    "average_epochs",
    "cluster_permutation_1samp",
    "eeg_adjacency",
    "evoked_difference",
    "grand_average",
    "group_erp",
]
