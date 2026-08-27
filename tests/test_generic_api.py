"""Tests for the generic (model-agnostic) API added in 0.3.1.

Covers the smoothing primitive (``smooth_field`` / ``Footprint.smoothed``),
``Footprint.level_for`` / ``captured_fraction``, the exported ``ALIASES``
table, and the ``TypeError`` on unsupported input containers.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import signal as sg

from fluxprint.footprint import (FFP_SMOOTH_KERNEL, Footprint, FootprintSeries,
                                 smooth_field)


def _gaussian(nx=41, ny=41, dx=10.0, sigma=60.0):
    """A unit-integral 2-D Gaussian footprint (analytic reference)."""
    x = (np.arange(nx) - (nx - 1) / 2) * dx
    y = (np.arange(ny) - (ny - 1) / 2) * dx
    xx, yy = np.meshgrid(x, y)
    f = np.exp(-(xx**2 + yy**2) / (2 * sigma**2)) / (2 * np.pi * sigma**2)
    return Footprint(f=f, x=x, y=y)


# --------------------------------------------------------------------------- #
# smooth_field / Footprint.smoothed                                           #
# --------------------------------------------------------------------------- #
def test_ffp_smooth_kernel_values():
    expected = np.array([[0.05, 0.1, 0.05],
                         [0.10, 0.4, 0.10],
                         [0.05, 0.1, 0.05]])
    assert np.array_equal(FFP_SMOOTH_KERNEL, expected)


def test_smooth_field_matches_reference_double_convolution():
    rng = np.random.default_rng(0)
    f = rng.random((21, 21))
    expected = sg.convolve2d(f, FFP_SMOOTH_KERNEL, mode="same")
    expected = sg.convolve2d(expected, FFP_SMOOTH_KERNEL, mode="same")
    assert np.array_equal(smooth_field(f), expected)


def test_smooth_field_zero_passes_is_identity():
    f = np.arange(9.0).reshape(3, 3)
    assert np.array_equal(smooth_field(f, passes=0), f)


def test_smooth_field_does_not_modify_input():
    f = np.arange(9.0).reshape(3, 3)
    original = f.copy()
    smooth_field(f)
    assert np.array_equal(f, original)


def test_footprint_smoothed_applies_kernel_and_stamps_attr():
    fp = _gaussian()
    out = fp.smoothed()
    assert np.array_equal(out.f, smooth_field(fp.f))
    assert out.attrs.get("smoothed") == 1
    # the original is untouched
    assert "smoothed" not in fp.attrs
    assert not np.array_equal(out.f, fp.f)


def test_series_aggregate_smoothing_unchanged_by_refactor():
    fp = _gaussian()
    series = FootprintSeries([fp._replace(time=1.0), fp._replace(time=2.0)])
    smoothed = series.aggregate(smooth=True)
    unsmoothed = series.aggregate(smooth=False)
    assert np.array_equal(smoothed.f, smooth_field(unsmoothed.f))


def test_utils_smooth_data_deprecated_and_delegates():
    utils = pytest.importorskip("fluxprint.utils")
    f = np.arange(25.0).reshape(5, 5)
    with pytest.warns(DeprecationWarning, match="smooth_field"):
        out = utils.smooth_data(f)
    assert np.array_equal(out, smooth_field(f))


# --------------------------------------------------------------------------- #
# Footprint.level_for / captured_fraction                                     #
# --------------------------------------------------------------------------- #
def test_level_for_analytic_gaussian():
    # For an isotropic 2-D Gaussian with peak p, the level enclosing a mass
    # fraction r is p * (1 - r).
    fp = _gaussian(nx=201, ny=201, dx=5.0, sigma=100.0)
    peak = float(fp.f.max())
    for r in (0.5, 0.8):
        assert fp.level_for(r) == pytest.approx(peak * (1 - r), rel=0.02)


def test_level_for_accepts_percentages():
    fp = _gaussian()
    assert fp.level_for(80) == fp.level_for(0.8)


def test_level_for_matches_contours_level():
    fp = _gaussian()
    contour = fp.contours(0.5)[0]
    assert fp.level_for(0.5) == contour["level"]


def test_level_for_out_of_range_raises():
    fp = _gaussian()
    with pytest.raises(ValueError, match=r"\(0, 0.9\]"):
        fp.level_for(0.95)
    with pytest.raises(ValueError, match=r"\(0, 0.9\]"):
        fp.level_for(0.0)


def test_level_for_unreachable_warns_and_returns_nan():
    # A domain capturing far less than 50% of the flux.
    fp = _gaussian(nx=5, ny=5, dx=5.0, sigma=1000.0)
    assert fp.captured_fraction < 0.5
    with pytest.warns(UserWarning, match="unreachable"):
        assert np.isnan(fp.level_for(0.5))


def test_captured_fraction_is_live_total():
    fp = _gaussian(nx=201, ny=201, dx=5.0, sigma=100.0)
    assert fp.captured_fraction == fp.total()
    assert 0.9 < fp.captured_fraction <= 1.0
    assert fp.normalized().captured_fraction == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# ALIASES                                                                     #
# --------------------------------------------------------------------------- #
def test_aliases_exported_at_top_level():
    import fluxprint
    assert fluxprint.ALIASES is fluxprint.core.ALIASES
    assert "ALIASES" in fluxprint.__all__
    assert fluxprint.ALIASES["model_inputs"]["ustar"] == ("u*",)
    assert "blh" in fluxprint.ALIASES["model_inputs"]["pblh"]
    assert fluxprint.ALIASES["drivers"]["H"] == ("H", "H_F")


def test_aliases_drive_column_resolution():
    import pandas as pd

    from fluxprint.core import process_footprint_inputs

    df = pd.DataFrame({
        "zm": [20.0], "z0": [0.1], "u*": [0.4], "blh": [900.0],
        "OL": [-50.0], "sigma_v": [0.4], "WD": [180.0],
    })
    inputs = process_footprint_inputs(data=df,
                                      estimate_missing_variables=False)
    assert inputs["ustar"] == [0.4]
    assert inputs["pblh"] == [900.0]
    assert inputs["mo_length"] == [-50.0]
    assert inputs["v_sigma"] == [0.4]
    assert inputs["wind_dir"] == [180.0]
