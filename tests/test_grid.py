"""Tests for fluxprint.grid: the hoisted FFP grid-resolution logic.

Every branch of ``resolve_grid`` is pinned to the concrete values the inline
Kljun/reference code produces (the bitwise gate for the field itself lives in
test_reference_regression.py — these pin the resolution arithmetic and axes).
"""
from __future__ import annotations

import numpy as np
import pytest

from fluxprint.grid import (GridContext, GridSpec, resolve_grid, rotate_theta,
                            to_wind_frame)


# --------------------------------------------------------------------------- #
# resolve_grid: one test per resolution branch                                #
# --------------------------------------------------------------------------- #
def test_default_branch_nothing_passed():
    spec = resolve_grid()
    assert spec == GridSpec(-1000., 1000., -1000., 1000., 2., 2., 1000, 1000)
    assert spec.shape == (1001, 1001)


def test_domain_with_dx_dy_branch_truncates_counts():
    spec = resolve_grid(domain=[-300, 300, -300, 300], dx=10.0, dy=10.0)
    assert (spec.nx, spec.ny) == (60, 60)
    assert spec.shape == (61, 61)
    x, y = spec.axes()
    assert np.array_equal(x, np.linspace(-300, 300, 61))
    assert np.array_equal(y, np.linspace(-300, 300, 61))
    # int() truncation, not rounding, on a non-divisible spacing:
    assert resolve_grid(domain=[-300, 300, -300, 300], dx=7.0).nx == 85


def test_domain_with_nx_branch():
    spec = resolve_grid(domain=[-200.0, 200.0, -200.0, 200.0], nx=150)
    assert (spec.nx, spec.ny) == (150, 150)
    assert spec.dx == 400.0 / 150.0 and spec.dy == 400.0 / 150.0


def test_domain_only_defaults_counts_to_1000():
    spec = resolve_grid(domain=[-100.0, 100.0, -50.0, 50.0])
    assert (spec.nx, spec.ny) == (1000, 1000)
    assert spec.dx == 0.2 and spec.dy == 0.1


def test_dx_and_nx_branch_sizes_the_domain():
    spec = resolve_grid(dx=15.0, nx=80)
    assert spec.domain == [-600.0, 600.0, -600.0, 600.0]
    assert (spec.dx, spec.dy, spec.nx, spec.ny) == (15.0, 15.0, 80, 80)


def test_dx_only_branch_uses_default_domain():
    spec = resolve_grid(dx=20.0)
    assert spec.domain == [-1000, 1000, -1000, 1000]
    assert (spec.nx, spec.ny) == (100, 100)
    assert spec.dy == 20.0


def test_nx_only_branch_uses_default_domain():
    spec = resolve_grid(nx=100)
    assert spec.domain == [-1000, 1000, -1000, 1000]
    assert spec.dx == 20.0 and spec.dy == 20.0


def test_squaring_up_partial_specs():
    assert resolve_grid(domain=[-100.0, 100.0, -100.0, 100.0], dy=5.0).dx == 5.0
    spec = resolve_grid(dx=15.0, ny=80)
    assert (spec.nx, spec.ny) == (80, 80)


def test_non_list_domain_is_ignored():
    # Verbatim reference quirk: a tuple domain is dropped, so dx-only rules apply.
    spec = resolve_grid(domain=(-300, 300, -300, 300), dx=10.0)
    assert spec.domain == [-1000, 1000, -1000, 1000]
    assert (spec.nx, spec.ny) == (200, 200)


# --------------------------------------------------------------------------- #
# GridContext / rotations                                                     #
# --------------------------------------------------------------------------- #
def test_grid_context_matches_inline_arrays():
    spec = resolve_grid(domain=[-300, 300, -300, 300], dx=10.0, dy=10.0)
    ctx = GridContext(spec)
    x = np.linspace(-300, 300, 61)
    x_2d, y_2d = np.meshgrid(x, x)
    assert np.array_equal(ctx.x_2d, x_2d)
    assert np.array_equal(ctx.y_2d, y_2d)
    assert np.array_equal(ctx.rho, np.sqrt(x_2d**2 + y_2d**2))
    assert np.array_equal(ctx.theta, np.arctan2(x_2d, y_2d))
    assert ctx.theta is ctx.theta  # cached, not recomputed


def test_rotate_theta_is_verbatim():
    theta = np.array([[0.0, np.pi / 2], [np.pi, -np.pi / 2]])
    assert np.array_equal(rotate_theta(theta, 90.0),
                          theta - 90.0 * np.pi / 180.)


def test_to_wind_frame_matches_polar_form():
    ctx = GridContext(resolve_grid(domain=[-300, 300, -300, 300], dx=10.0))
    for wind_dir in (0.0, 30.0, 137.0, 270.0):
        along, cross = to_wind_frame(ctx.x_2d, ctx.y_2d, wind_dir)
        rot = rotate_theta(ctx.theta, wind_dir)
        assert np.allclose(along, ctx.rho * np.cos(rot), atol=1e-9)
        assert np.allclose(cross, ctx.rho * np.sin(rot), atol=1e-9)


def test_along_wind_points_at_the_wind_source():
    # Wind from the east (90 deg): the fetch is east of the tower, so a point
    # east of the tower has a positive along-wind coordinate.
    along, cross = to_wind_frame(np.array([[100.0]]), np.array([[0.0]]), 90.0)
    assert along[0, 0] == pytest.approx(100.0)
    assert cross[0, 0] == pytest.approx(0.0, abs=1e-9)
