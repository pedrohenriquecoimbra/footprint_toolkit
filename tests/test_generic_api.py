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
