"""Tests for p0ly_utils.analysis.cfc (US-027).

All data is simulated in memory (small grids, few surrogates) so the fast
suite stays small — the full null arrays are always retained but tiny.

Synthetic layout: 2 EEG channels at 250 Hz, 1000-sample (4 s) epochs.
Channel 0 carries a 6 Hz rhythm whose cosine modulates the amplitude of a
25 Hz carrier (PAC) and a 2:1 harmonic relation (PPC); channel 1 carries the
same low-frequency rhythm with an unmodulated / phase-independent high band
(control). Per-epoch phase offsets keep the signal non-pathological —
identical phases in every epoch make circular-shift surrogates
indistinguishable from the observed map.

Frequency selectivity is asserted on the amplitude axis and across channels:
tensorpac's Hilbert path uses a fixed-cycle FIR whose transition band at low
frequencies is wide, so a 6 Hz rhythm leaks into neighbouring *phase* bands
by design of the dependency, not of this wrapper.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from mne import EpochsArray, create_info

from p0ly_utils.analysis import (
    CFCMap,
    SubjectCFCResult,
    cfc_condition_contrast,
    estimate_cfc,
    group_cfc,
)

_SFREQ = 250.0
_N_TIMES = 1000  # 4 s
_N_EPOCHS = 8
_CH_NAMES = ("Fp1", "Fp2")
_PHASE_FREQS = np.array([6.0, 10.0])
_AMP_FREQS = np.array([15.0, 25.0, 35.0])
_SHAPE = (_CH_NAMES.__len__(), _PHASE_FREQS.size, _AMP_FREQS.size)
_PHASE_BW = 2.0
_AMP_BW = 8.0
_LO_HZ = 6.0
_HI_HZ = 25.0
_SEED = 20261020
_PAC_CELL = (0, 1)  # (phase 6 Hz, amp 25 Hz) indices into the grids
_PPC_CELL = (1, 1)  # (phase 6 Hz, amp 12 Hz) on the dedicated PPC grid


def _epochs(*, n_epochs: int = _N_EPOCHS) -> EpochsArray:
    """Two-channel epochs; channel 0 inherits the injected coupling only."""
    rng = np.random.default_rng(_SEED)
    times = np.arange(_N_TIMES) / _SFREQ
    data = np.zeros((n_epochs, 2, _N_TIMES))
    for epoch in range(n_epochs):
        low = 2.0 * np.pi * _LO_HZ * times + rng.uniform(0.0, 2.0 * np.pi)
        high = 2.0 * np.pi * _HI_HZ * times + rng.uniform(0.0, 2.0 * np.pi)
        data[epoch, 0] += np.sin(low) + (1.0 + 0.9 * np.cos(low)) * np.sin(high)
        data[epoch, 1] += np.sin(low) + np.sin(high)
    data += rng.normal(size=data.shape) * 0.1
    return EpochsArray(data, create_info(list(_CH_NAMES), _SFREQ, "eeg"))


def _ppc_epochs(n_epochs: int = 16) -> EpochsArray:
    """Epochs with a stable 2:1 harmonic (ch 0) and jittered high phase (ch 1).

    The control channel's independent high-band phase cancels across trials;
    the residual mean resultant shrinks like ``1 / sqrt(n_epochs)``, which is
    why this fixture uses more trials than the PAC one.
    """
    rng = np.random.default_rng(_SEED)
    times = np.arange(_N_TIMES) / _SFREQ
    data = np.zeros((n_epochs, 2, _N_TIMES))
    for epoch in range(n_epochs):
        low = 2.0 * np.pi * _LO_HZ * times + rng.uniform(0.0, 2.0 * np.pi)
        independent = 2.0 * np.pi * 2.0 * _LO_HZ * times + rng.uniform(0.0, 2.0 * np.pi)
        data[epoch, 0] += np.sin(low) + np.sin(2.0 * low)
        data[epoch, 1] += np.sin(low) + np.sin(independent)
    data += rng.normal(size=data.shape) * 0.05
    return EpochsArray(data, create_info(list(_CH_NAMES), _SFREQ, "eeg"))


def _ppc_grid() -> tuple[np.ndarray, np.ndarray]:
    """Grids where a 2:1 ratio lands on exact cells (6 -> 12, 8 -> 16 Hz)."""
    return np.array([4.0, 6.0, 8.0]), np.array([8.0, 12.0, 16.0])


def _ratio_key(n: int, m: int) -> str:
    return f"ppc_{n}x{m}"


def _estimate(**overrides: object):
    """Estimate with valid PAC-only defaults; ``overrides`` replaces keywords."""
    kwargs: dict[str, object] = {
        "methods": ["pac"],
        "phase_freqs": _PHASE_FREQS,
        "amp_freqs": _AMP_FREQS,
        "phase_bandwidth": _PHASE_BW,
        "amp_bandwidth": _AMP_BW,
        "n_surrogates": 2,
        "seed": _SEED,
    }
    epochs = overrides.pop("epochs", _epochs())
    kwargs.update(overrides)
    return estimate_cfc(epochs, **kwargs)  # type: ignore[arg-type]


def _ppc_estimate(**overrides: object):
    """Estimate with valid PPC-only defaults on the ratio-compatible grid."""
    phase_freqs, amp_freqs = _ppc_grid()
    kwargs: dict[str, object] = {
        "methods": ["phase_phase"],
        "phase_freqs": phase_freqs,
        "amp_freqs": amp_freqs,
        "phase_bandwidth": _PHASE_BW,
        "amp_bandwidth": _AMP_BW,
        "ratios": [(2, 1)],
        "ratio_tolerance": 0.6,
        "n_surrogates": 3,
        "seed": _SEED,
    }
    epochs = overrides.pop("epochs", _ppc_epochs())
    kwargs.update(overrides)
    return estimate_cfc(epochs, **kwargs)  # type: ignore[arg-type]


def _cfc_map(
    values: np.ndarray,
    *,
    method: str = "pac",
    valid_mask: np.ndarray | None = None,
) -> CFCMap:
    """Hand-built observed map (keeps estimator runtime out of cohort tests)."""
    return CFCMap(
        values=np.asarray(values, dtype=float),
        ch_names=_CH_NAMES,
        phase_freqs=_PHASE_FREQS.copy(),
        amp_freqs=_AMP_FREQS.copy(),
        method=method,
        n_epochs=10,
        valid_mask=(
            np.ones(np.shape(values), dtype=bool)
            if valid_mask is None
            else np.asarray(valid_mask, dtype=bool)
        ),
    )


# --- PAC --------------------------------------------------------------------


@pytest.fixture(scope="module")
def pac_results() -> dict[str, SubjectCFCResult]:
    """Observed/null for both probe-verified PAC estimators."""
    return {
        "gc": _estimate(n_surrogates=8)["pac"],
        "mvl": _estimate(pac_method="mean_vector_length", n_surrogates=4)["pac"],
    }


def test_pac_map_null_and_pvalue_contract(pac_results: dict[str, SubjectCFCResult]) -> None:
    """AC: (n_channels, n_phase_freqs, n_amp_freqs) maps with a full null."""
    result = pac_results["gc"]
    observed = result.observed
    assert observed.values.shape == _SHAPE
    assert result.null.shape == (8,) + _SHAPE
    assert result.p_values.shape == _SHAPE
    assert observed.ch_names == _CH_NAMES
    assert np.array_equal(observed.phase_freqs, _PHASE_FREQS)
    assert np.array_equal(observed.amp_freqs, _AMP_FREQS)
    assert observed.method == "pac"
    assert observed.n_epochs == _N_EPOCHS
    assert observed.valid_mask.all()
    assert result.evidence_level == "subject"
    assert result.n_surrogates == 8 and result.seed == _SEED
    assert np.all(np.isfinite(observed.values)) and np.all(np.isfinite(result.null))
    assert np.all((result.p_values > 0.0) & (result.p_values <= 1.0))


def test_pac_recovers_injected_coupling(pac_results: dict[str, SubjectCFCResult]) -> None:
    """Injected cell beats the other amplitude bins and the control channel."""
    values = pac_results["gc"].observed.values
    target = values[0][_PAC_CELL]
    assert target > 0.3
    row = values[0][_PAC_CELL[0]]
    assert target > 3.0 * np.delete(row, _PAC_CELL[1]).max()  # amp-axis specificity
    assert target > 3.0 * values[1].max()  # control channel


def test_pac_axis_orientation(pac_results: dict[str, SubjectCFCResult]) -> None:
    """Axes are (channel, phase, amplitude) — tensorpac's order is transposed."""
    values = pac_results["gc"].observed.values
    # Unequal grid sizes (2 phase bins, 3 amplitude bins) make a wrong
    # transpose show up as a shape mismatch in the contract test; the argmax
    # additionally pins the injected (phase 6 Hz, amp 25 Hz) cell.
    assert values.shape == (len(_CH_NAMES), 2, 3)
    assert np.unravel_index(np.argmax(values[0]), values[0].shape) == _PAC_CELL
    assert values[1][_PAC_CELL] < 0.05


def test_pac_mean_vector_length_runs(pac_results: dict[str, SubjectCFCResult]) -> None:
    """Verified MVL ID yields finite non-negative coupling, weaker than gcPAC."""
    result = pac_results["mvl"]
    assert np.all(np.isfinite(result.observed.values))
    assert np.all(result.observed.values >= 0.0)
    assert result.observed.values[0][_PAC_CELL] > result.observed.values[1][_PAC_CELL]


def test_pac_null_is_reproducible_and_seed_sensitive() -> None:
    """Same data/config/seed reproduces observed *and* null exactly."""
    first = _estimate(n_surrogates=4)["pac"]
    second = _estimate(n_surrogates=4)["pac"]
    other = _estimate(n_surrogates=4, seed=_SEED + 1)["pac"]
    assert np.array_equal(first.observed.values, second.observed.values)
    assert np.array_equal(first.null, second.null)
    assert not np.array_equal(first.null, other.null)


# --- n:m phase-phase coupling ------------------------------------------------


@pytest.fixture(scope="module")
def ppc_result():
    """Observed/null PPC for a 2:1 ratio on a ratio-compatible grid."""
    return _ppc_estimate(n_surrogates=6)[_ratio_key(2, 1)]


def test_ppc_recovers_injected_harmonic(ppc_result) -> None:
    """The 2:1 relation shows up at (6 Hz, 12 Hz) and stays in [0, 1]."""
    values = ppc_result.observed.values
    target = values[0][_PPC_CELL]
    assert 0.0 <= target <= 1.0
    assert target > 0.5
    assert target > 3.0 * values[1][_PPC_CELL]  # control channel


def test_ppc_ratio_mask_and_nan_off_ratio(ppc_result) -> None:
    """Only ratio-compatible cells are valid; off-ratio cells are NaN."""
    mask = ppc_result.observed.valid_mask
    expected = np.zeros(mask.shape, dtype=bool)
    for phase_index, center in enumerate(ppc_result.observed.phase_freqs):
        for amp_index, amp_center in enumerate(ppc_result.observed.amp_freqs):
            expected[:, phase_index, amp_index] = abs(amp_center - 2.0 * center) <= 0.6
    assert np.array_equal(mask, expected)
    assert np.all(np.isnan(ppc_result.observed.values[~mask]))
    assert np.all(np.isfinite(ppc_result.observed.values[mask]))
    assert np.all(np.isnan(ppc_result.p_values[~mask]))
    assert np.all(np.isnan(ppc_result.null[:, ~mask]))
    assert ppc_result.observed.method == _ratio_key(2, 1)
    assert ppc_result.null.shape == (6,) + ppc_result.observed.values.shape
    valid_p = ppc_result.p_values[mask]
    assert np.all((valid_p > 0.0) & (valid_p <= 1.0))
    # Valid cells are the exact (4 -> 8), (6 -> 12) and (8 -> 16) diagonals.
    assert mask.sum() == 3 * len(_CH_NAMES)


def test_ppc_is_reproducible(ppc_result) -> None:
    """Same seed reproduces the PPC null and observed maps."""
    repeat = _ppc_estimate(n_surrogates=6)[_ratio_key(2, 1)]
    assert np.array_equal(repeat.observed.values, ppc_result.observed.values, equal_nan=True)
    assert np.array_equal(repeat.null, ppc_result.null, equal_nan=True)


def test_ppc_uncoupled_control_stays_bounded() -> None:
    """Uncoupled noise: valid coupling is finite and inside [0, 1]."""
    rng = np.random.default_rng(7)
    noise = EpochsArray(
        rng.normal(size=(4, 2, _N_TIMES)) * 1e-6,
        create_info(list(_CH_NAMES), _SFREQ, "eeg"),
    )
    result = _ppc_estimate(epochs=noise)[_ratio_key(2, 1)]
    valid = result.observed.valid_mask
    assert np.all((result.observed.values[valid] >= 0.0) & (result.observed.values[valid] <= 1.0))
    assert np.all(result.p_values[valid] > 0.0)


def test_both_methods_share_one_call() -> None:
    """PAC and every PPC ratio come back keyed from a single estimate call."""
    phase_freqs, amp_freqs = _ppc_grid()
    results = estimate_cfc(
        _epochs(n_epochs=4),
        methods=["pac", "phase_phase"],
        phase_freqs=phase_freqs,
        amp_freqs=amp_freqs,
        phase_bandwidth=_PHASE_BW,
        amp_bandwidth=_AMP_BW,
        ratios=[(2, 1), (3, 1)],
        ratio_tolerance=0.6,
        n_surrogates=3,
        seed=_SEED,
    )
    assert list(results) == ["pac", "ppc_2x1", "ppc_3x1"]
    assert all(result.evidence_level == "subject" for result in results.values())


# --- guards ------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"methods": []}, id="no-methods"),
        pytest.param({"methods": ["plv"]}, id="unknown-method"),
        pytest.param({"pac_method": "modulation_index"}, id="unknown-pac-method"),
        pytest.param({"phase_freqs": np.array([6.0])}, id="single-phase-bin"),
        pytest.param({"phase_freqs": np.array([10.0, 6.0])}, id="unsorted-grid"),
        pytest.param({"phase_freqs": np.array([6.0, np.nan])}, id="non-finite-grid"),
        pytest.param({"phase_bandwidth": 0.0}, id="zero-bandwidth"),
        pytest.param({"phase_bandwidth": -2.0}, id="negative-bandwidth"),
        pytest.param({"phase_freqs": np.array([1.0, 2.0]), "phase_bandwidth": 4.0}, id="dc-edge"),
        pytest.param({"amp_freqs": np.array([120.0, 124.0]), "amp_bandwidth": 8.0}, id="nyquist"),
        pytest.param({"n_surrogates": 0}, id="zero-surrogates"),
        pytest.param({"n_surrogates": 2.5}, id="float-surrogates"),
        pytest.param({"ratio_tolerance": 0.0}, id="zero-tolerance"),
    ],
)
def test_invalid_configuration_fails_loud(overrides: dict[str, object]) -> None:
    """Every invalid grid/band/count raises ValueError before any estimation."""
    with pytest.raises(ValueError):
        _estimate(**overrides)


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"ratios": []}, id="no-ratios"),
        pytest.param({"ratios": [(4, 2)]}, id="reducible-ratio"),
        pytest.param({"ratios": [(0, 1)]}, id="zero-ratio"),
        pytest.param({"ratios": [(-2, 1)]}, id="negative-ratio"),
        pytest.param({"ratios": [(1.5, 1)]}, id="float-ratio"),
        pytest.param({"ratios": [(5, 1)], "ratio_tolerance": 0.1}, id="no-valid-cell"),
    ],
)
def test_invalid_ratio_fails_loud(overrides: dict[str, object]) -> None:
    """PPC ratios must be positive coprime integers covering >= 1 valid cell."""
    with pytest.raises(ValueError):
        _ppc_estimate(**overrides)


def test_too_few_epochs_fails() -> None:
    """Cross-frequency coupling is an across-trials statistic (need >= 2)."""
    with pytest.raises(ValueError, match="need >= 2"):
        _estimate(epochs=_epochs(n_epochs=1))


def test_too_few_samples_fails() -> None:
    """Epochs shorter than the SOS padding floor fail loud, not silently."""
    short = EpochsArray(np.zeros((4, 2, 20)), create_info(list(_CH_NAMES), _SFREQ, "eeg"))
    with pytest.raises(ValueError, match="samples"):
        _estimate(epochs=short)


def test_non_finite_data_fails() -> None:
    """NaN samples are rejected instead of silently poisoning the null."""
    epochs = _epochs()
    epochs._data[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        _estimate(epochs=epochs)


def test_condition_query_selects_subset() -> None:
    """A metadata query narrows the epoch set; zero matches fail loud."""
    epochs = _epochs(n_epochs=6)
    epochs.metadata = pd.DataFrame({"Correct": [True, False, True, False, True, False]})
    result = _estimate(epochs=epochs, condition_query="Correct == True")["pac"]
    assert result.observed.n_epochs == 3
    with pytest.raises(ValueError, match="need >= 2"):
        _estimate(epochs=epochs, condition_query="Correct > 5")


# --- descriptive cohort maps -------------------------------------------------


def test_group_cfc_single_subject_is_identity() -> None:
    """n=1 is allowed and returns the observed values unchanged."""
    values = np.arange(np.prod(_SHAPE), dtype=float).reshape(_SHAPE) / 100.0
    group = group_cfc([{"pac": _cfc_map(values)}])
    assert group.n_subjects == 1
    assert group.evidence_level == "cohort_descriptive"
    assert np.array_equal(group.averages["pac"].values, values)
    assert group.averages["pac"].n_epochs is None


def test_group_cfc_equal_weight_mean_ignores_trial_counts() -> None:
    """Means are arithmetic over subjects even with unequal epoch counts."""
    base = np.zeros(_SHAPE)
    first, second, third = _cfc_map(base + 0.2), _cfc_map(base + 0.6), _cfc_map(base + 0.4)
    second.n_epochs = 500
    third.n_epochs = 3
    group = group_cfc([{"pac": first}, {"pac": second}, {"pac": third}])
    assert group.n_subjects == 3
    assert np.allclose(group.averages["pac"].values, base + 0.4)


def test_group_cfc_preserves_off_ratio_nan() -> None:
    """A shared PPC mask keeps NaN off-ratio cells through averaging."""
    mask = np.zeros(_SHAPE, dtype=bool)
    mask[:, 1, 1] = True
    values = np.full(_SHAPE, np.nan)
    values[mask] = 0.5
    other = np.full(_SHAPE, np.nan)
    other[mask] = 0.7
    key = _ratio_key(2, 1)
    group = group_cfc(
        [
            {key: _cfc_map(values, method=key, valid_mask=mask)},
            {key: _cfc_map(other, method=key, valid_mask=mask)},
        ]
    )
    averaged = group.averages[key]
    assert np.array_equal(averaged.valid_mask, mask)
    assert np.allclose(averaged.values[mask], 0.6)
    assert np.all(np.isnan(averaged.values[~mask]))


def test_group_cfc_empty_input_fails() -> None:
    with pytest.raises(ValueError, match="no subjects"):
        group_cfc([])


@pytest.mark.parametrize("mutation", ["channels", "grid", "mask", "method"])
def test_group_cfc_mismatch_fails(mutation: str) -> None:
    """Channel/grid/method/mask mismatches fail loud (US-024 philosophy)."""
    reference = _cfc_map(np.zeros(_SHAPE))
    other = _cfc_map(np.zeros(_SHAPE))
    key = "pac"
    if mutation == "channels":
        other.ch_names = ("Fp1", "Oz")
    elif mutation == "grid":
        other.phase_freqs = np.array([6.0, 10.5])
    elif mutation == "mask":
        mask = np.ones(_SHAPE, dtype=bool)
        mask[0, 0, 0] = False
        other.valid_mask = mask
    else:
        key = _ratio_key(2, 1)
    with pytest.raises(ValueError):
        group_cfc([{"pac": reference}, {key: other}])


def test_group_cfc_rejects_subject_results() -> None:
    """Cohort math accepts observed CFCMap objects only — no subject nulls."""
    result = _estimate()["pac"]
    with pytest.raises(TypeError, match="observed CFCMap"):
        group_cfc([{"pac": result}])  # type: ignore[dict-item]


# --- cohort condition contrast ----------------------------------------------


def _arm(
    rows: np.ndarray,
    *,
    method: str = "pac",
    valid_mask: np.ndarray | None = None,
) -> list[dict[str, CFCMap]]:
    """One subject mapping per row of ``rows``."""
    return [{method: _cfc_map(row, method=method, valid_mask=valid_mask)} for row in rows]


def test_contrast_direction_is_first_minus_second() -> None:
    """obs = mean(first - second) at the subject unit, per cell."""
    first = _arm(np.full((2,) + _SHAPE, 0.4))
    second = _arm(np.full((2,) + _SHAPE, 0.1))
    result = cfc_condition_contrast(
        first, second, conditions=("a", "b"), n_permutations=16, seed=_SEED
    )["pac"]
    assert np.allclose(result.difference.values, 0.3)
    assert result.conditions == ("a", "b")
    assert result.correction == "full_map_max_abs_mean"
    assert result.evidence_level == "cohort_subject_unit"
    assert result.n_subjects == 2
    assert result.difference.method == "pac"
    assert result.difference.n_epochs is None
    assert result.difference.ch_names == _CH_NAMES


def test_contrast_exact_enumeration_counts_and_resolution() -> None:
    """Exact path enumerates 2 ** (n - 1) patterns; small n cannot reach 0.05."""
    first = _arm(np.zeros((3,) + _SHAPE))
    second = _arm(np.zeros((3,) + _SHAPE))
    result = cfc_condition_contrast(
        first, second, conditions=("a", "b"), n_permutations=1024, seed=_SEED
    )["pac"]
    assert result.n_resamples == 2 ** (3 - 1) == 4
    assert result.null_max.shape == (4,)
    # Constant maps: every resample hits the observed |max| -> p = (1 + 4) / 5.
    assert np.allclose(result.p_values_corrected, (1.0 + 4.0) / (4.0 + 1.0))
    # n=2 fixes one resolution floor, documented as 1/3 with the +1 convention.
    pair = cfc_condition_contrast(
        first[:2], second[:2], conditions=("a", "b"), n_permutations=1024, seed=_SEED
    )["pac"]
    assert pair.n_resamples == 2
    assert np.all(pair.p_values_corrected >= 1.0 / 3.0)


def test_contrast_random_path_length_and_reproducibility() -> None:
    """Above the exact cap, n_permutations random patterns are drawn and seeded."""
    rng = np.random.default_rng(11)
    first = _arm(rng.normal(size=(8,) + _SHAPE))
    second = _arm(rng.normal(size=(8,) + _SHAPE))
    result = cfc_condition_contrast(
        first, second, conditions=("a", "b"), n_permutations=16, seed=_SEED
    )["pac"]
    repeat = cfc_condition_contrast(
        first, second, conditions=("a", "b"), n_permutations=16, seed=_SEED
    )["pac"]
    assert result.n_resamples == 16
    assert result.null_max.shape == (16,)
    assert np.array_equal(result.null_max, repeat.null_max)
    assert np.array_equal(result.p_values_corrected, repeat.p_values_corrected)
    other = cfc_condition_contrast(
        first, second, conditions=("a", "b"), n_permutations=16, seed=_SEED + 1
    )["pac"]
    assert not np.array_equal(result.null_max, other.null_max)
    assert np.all((result.p_values_corrected > 0.0) & (result.p_values_corrected <= 1.0))


def test_contrast_detects_consistent_difference_over_noise() -> None:
    """A uniform subject-level shift beats a noisy cell, at small-n resolution."""
    injected = np.zeros(_SHAPE)
    injected[0, 1, 1] = 0.5  # consistent condition difference on every subject
    rng = np.random.default_rng(3)
    noise = rng.normal(scale=0.02, size=(6,) + _SHAPE)
    first = _arm(injected[np.newaxis] + noise)
    second = _arm(np.zeros((6,) + _SHAPE))
    result = cfc_condition_contrast(
        first, second, conditions=("a", "b"), n_permutations=32, seed=_SEED
    )["pac"]
    assert result.difference.values[0, 1, 1] > 0.4
    assert result.p_values_corrected[0, 1, 1] < result.p_values_corrected[0, 0, 0]
    assert result.p_values_corrected[0, 1, 1] <= 2.0 / (result.n_resamples + 1.0)


def test_contrast_keeps_invalid_cells_nan_and_excludes_them() -> None:
    """Off-ratio cells stay NaN and never join the correction family."""
    mask = np.zeros(_SHAPE, dtype=bool)
    mask[:, 1, 1] = True
    values = np.full(_SHAPE, np.nan)
    values[mask] = 0.3
    key = _ratio_key(2, 1)
    first = _arm(np.stack([values, values * 2.0]), method=key, valid_mask=mask)
    second = _arm(np.stack([values * 0.5, values * 0.5]), method=key, valid_mask=mask)
    result = cfc_condition_contrast(
        first, second, conditions=("a", "b"), n_permutations=8, seed=_SEED
    )[key]
    assert np.all(np.isnan(result.p_values_corrected[~mask]))
    assert np.all(np.isfinite(result.p_values_corrected[mask]))
    assert np.all(np.isnan(result.difference.values[~mask]))
    assert np.array_equal(result.difference.valid_mask, mask)


@pytest.mark.parametrize(
    "mutation", ["too-few", "count-mismatch", "channels", "grid", "mask", "method"]
)
def test_contrast_guards_fail_loud(mutation: str) -> None:
    """n<2, arm length, and method/channel/grid/mask mismatches all raise."""
    zeros = np.zeros((3,) + _SHAPE)
    first, second = _arm(zeros), _arm(zeros)
    match = ""
    if mutation == "too-few":
        first, second = first[:1], second[:1]
        match = "needs >= 2"
    elif mutation == "count-mismatch":
        second = second[:2]
        match = "arm sizes differ"
    elif mutation == "channels":
        second[0]["pac"].ch_names = ("Fp1", "Oz")
        match = "channel mismatch"
    elif mutation == "grid":
        second[0]["pac"].amp_freqs = np.array([15.0, 25.0, 36.0])
        match = "amp_freqs grid mismatch"
    elif mutation == "mask":
        second[0]["pac"].valid_mask = np.ones(_SHAPE, dtype=bool)
        second[0]["pac"].valid_mask[0, 0, 0] = False
        match = "valid_mask mismatch"
    else:
        second = _arm(zeros, method=_ratio_key(2, 1))
        match = "method mismatch"
    with pytest.raises(ValueError, match=match):
        cfc_condition_contrast(first, second, conditions=("a", "b"), seed=_SEED)


def test_contrast_rejects_subject_results() -> None:
    """Subject p-values/nulls cannot enter cohort inference by construction."""
    result = _estimate()["pac"]
    with pytest.raises(TypeError, match="observed CFCMap"):
        cfc_condition_contrast(
            [{"pac": result}, {"pac": result}],  # type: ignore[dict-item]
            [{"pac": result}, {"pac": result}],  # type: ignore[dict-item]
            conditions=("a", "b"),
            seed=_SEED,
        )


def test_contrast_signature_has_no_pvalue_input() -> None:
    """The contrast API has no subject p-value/null parameter at all."""
    import inspect

    parameters = inspect.signature(cfc_condition_contrast).parameters
    assert set(parameters) == {"first", "second", "conditions", "n_permutations", "seed"}
