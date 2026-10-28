"""Analysis modules for p0ly_utils (US-010+): ERP, PSD, TFR, connectivity, ...

Each analysis domain lives in its own submodule. All functions are pure
(stateless, no filesystem paths) and documented with array shapes per
[[SCHEMA]] §1.
"""

from p0ly_utils.analysis.cfc import (
    CFCContrastResult,
    CFCMap,
    GroupCFCResult,
    SubjectCFCResult,
    cfc_condition_contrast,
    estimate_cfc,
    group_cfc,
)
from p0ly_utils.analysis.connectivity import (
    ConnectivityMatrices,
    GraphMetrics,
    band_average,
    difference_connectivity,
    graph_metrics,
    group_connectivity,
    spectral_connectivity,
)
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
from p0ly_utils.analysis.psd import (
    AperiodicFit,
    average_spectrum,
    band_powers,
    fit_aperiodic,
    fit_aperiodic_channels,
    group_psd,
    psd_by_condition,
    psd_epochs,
    psd_raw,
)
from p0ly_utils.analysis.regression import (
    GroupRegressionResult,
    RegressionResult,
    group_regression,
    regression_epochs,
)
from p0ly_utils.analysis.tfr import (
    average_power,
    freq_grid,
    group_tfr,
    itpc,
    itpc_z,
    tfr_cluster_permutation,
    tfr_difference,
    tfr_epochs,
    tfr_power_epochs,
)

__all__ = [
    "AperiodicFit",
    "CFCMap",
    "CFCContrastResult",
    "ClusterPermStats",
    "ConnectivityMatrices",
    "GraphMetrics",
    "GroupCFCResult",
    "GroupERPResult",
    "GroupRegressionResult",
    "RegressionResult",
    "SubjectCFCResult",
    "average_epochs",
    "average_power",
    "average_spectrum",
    "band_average",
    "band_powers",
    "cfc_condition_contrast",
    "estimate_cfc",
    "freq_grid",
    "difference_connectivity",
    "cluster_permutation_1samp",
    "eeg_adjacency",
    "evoked_difference",
    "fit_aperiodic",
    "fit_aperiodic_channels",
    "grand_average",
    "group_erp",
    "group_connectivity",
    "group_cfc",
    "group_psd",
    "group_regression",
    "group_tfr",
    "graph_metrics",
    "itpc",
    "itpc_z",
    "psd_by_condition",
    "psd_epochs",
    "psd_raw",
    "regression_epochs",
    "spectral_connectivity",
    "tfr_cluster_permutation",
    "tfr_difference",
    "tfr_epochs",
    "tfr_power_epochs",
]
