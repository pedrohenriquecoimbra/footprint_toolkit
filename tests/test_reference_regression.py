"""Golden regression tests pinning every registered model to its reference.

The package vendors the original FFP code (fluxprint/model/Kljun_et_al_2015_original)
precisely so the rewritten port can be validated against it. These tests keep
the two EXACTLY identical (bitwise, not merely close): any edit to the port
that changes the field fails here, and CI runs this file as its own named job.

Adding a model? Register a reference oracle in ``REFERENCE_ORACLES`` below —
``test_every_registered_model_has_a_reference_oracle`` fails until you do.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("rasterio")  # the package import pulls the geo stack

from fluxprint.model import available_models, get_model  # noqa: E402
from fluxprint.model import Kljun_et_al_2015 as port  # noqa: E402
from fluxprint.model.Kljun_et_al_2015_original import (  # noqa: E402
    calc_footprint_FFP_climatology as reference)

GRID = dict(domain=[-300, 300, -300, 300], dx=10.0, dy=10.0)

#: Canonical input battery: one case per stability regime, both wind-speed
#: modes, and a multi-record composite. Every oracle-checked model runs all.
CASES = {
    "unstable-z0": dict(zm=[20.0], z0=[0.1], ustar=[0.5], pblh=[1000.0],
                        mo_length=[-100.0], v_sigma=[0.5], wind_dir=[30.0]),
    "neutral-z0": dict(zm=[20.0], z0=[0.1], ustar=[0.5], pblh=[1000.0],
                       mo_length=[1.0e6], v_sigma=[0.5], wind_dir=[30.0]),
    "stable-z0": dict(zm=[20.0], z0=[0.1], ustar=[0.5], pblh=[1000.0],
                      mo_length=[200.0], v_sigma=[0.5], wind_dir=[30.0]),
    "unstable-umean": dict(zm=[20.0], umean=[3.0], ustar=[0.5], pblh=[1000.0],
                           mo_length=[-100.0], v_sigma=[0.5], wind_dir=[30.0]),
    "composite": dict(zm=[20.0] * 3, z0=[0.1] * 3, ustar=[0.5, 0.4, 0.3],
                      pblh=[1000.0] * 3, mo_length=[-100.0, -50.0, 200.0],
                      v_sigma=[0.5] * 3, wind_dir=[30.0, 180.0, 270.0]),
}


def _kljun2015_reference(**met):
    """Reference field for the canonical kwargs, via the vendored FFP code."""
    out = reference.FFP_climatology(
        zm=met["zm"], z0=met.get("z0"), umean=met.get("umean"),
        ustar=met["ustar"], h=met["pblh"], ol=met["mo_length"],
        sigmav=met["v_sigma"], wind_dir=met["wind_dir"],
        rs=None, verbosity=0, fig=False, **GRID)
    return np.asarray(out["fclim_2d"]), int(out["n"])


_HSIEH_SNAPSHOT = Path(__file__).parent / "data" / "hsieh2000_reference.npz"


def _hsieh2000_reference(**met):
    """Golden-snapshot oracle for the Hsieh port (a regression pin).

    There is no vendored original for this model; the snapshot was generated
    by ``tests/data/regenerate_hsieh_reference.py`` at registration time and
    pins the field thereafter. The tie to the published equations lives in
    ``tests/test_hsieh_reference.py`` (analytic invariants).
    """
    case = next(name for name, kwargs in CASES.items() if kwargs == met)
    with np.load(_HSIEH_SNAPSHOT) as data:
        return data[f"{case}_f"], int(data[f"{case}_n"])


#: model name -> oracle returning ``(field, n)`` for the canonical kwargs.
#: EVERY registered model must have an entry; the guard test enforces it.
REFERENCE_ORACLES = {
    "kljun2015": _kljun2015_reference,
    "hsieh2000": _hsieh2000_reference,
}

#: Canonical cases a model legitimately cannot run (skipped, not failed).
UNSUPPORTED_CASES = {
    "hsieh2000": {"unstable-umean"},  # Hsieh has no umean mode (z0 required)
}

#: Comparison tolerance per oracle. 0.0 = bitwise (the Kljun oracle runs the
#: vendored reference in the same process, so exact equality is well
#: defined). The Hsieh oracle is a snapshot generated on one platform;
#: libm/numpy exp/pow may differ by a few ulps elsewhere, so it gets a
#: tolerance far below any physical change but above ulp noise.
ORACLE_RTOL = {
    "kljun2015": 0.0,
    "hsieh2000": 1e-12,
}


def test_every_registered_model_has_a_reference_oracle():
    """A model without a reference check must not ship: register an oracle."""
    missing = [name for name in available_models()
               if name not in REFERENCE_ORACLES]
    assert not missing, (
        f"Registered model(s) {missing} have no reference oracle in "
        "tests/test_reference_regression.py::REFERENCE_ORACLES - every model "
        "must be pinned against its original implementation.")


@pytest.mark.parametrize("case", sorted(CASES))
@pytest.mark.parametrize("name", sorted(REFERENCE_ORACLES))
def test_registered_model_matches_reference_exactly(name, case):
    """The registered model's field is bitwise identical to its reference."""
    if name not in available_models():
        pytest.skip(f"{name} not registered in this build")
    if case in UNSUPPORTED_CASES.get(name, ()):
        pytest.skip(f"{name} does not support the {case} input mode")
    met = CASES[case]
    fp = get_model(name)(**{k: v for k, v in met.items()}, **GRID)
    expected, n = REFERENCE_ORACLES[name](**met)
    assert fp.n == n, f"{name}/{case}: composited record count differs"
    assert fp.f.dtype == expected.dtype, (
        f"{name}/{case}: field dtype differs from the reference")
    rtol = ORACLE_RTOL[name]
    matches = (np.array_equal(fp.f, expected) if rtol == 0.0
               else np.allclose(fp.f, expected, rtol=rtol, atol=0.0))
    assert matches, (
        f"{name}/{case}: field differs from the reference "
        f"(max abs diff {np.nanmax(np.abs(fp.f - expected)):.3e})")


#: Grid-spec variants covering every branch of the domain/dx/dy/nx/ny
#: resolution logic. The CASES battery above pins only the domain+dx+dy
#: branch; any hoist of the grid code must keep all branches bitwise
#: identical, axes included.
GRID_VARIANTS = {
    "default": {},
    "nx-only": dict(nx=100),
    "dx-only": dict(dx=20.0),
    "domain-nx": dict(domain=[-200.0, 200.0, -200.0, 200.0], nx=150),
    "dx-nx": dict(dx=15.0, nx=80),
}


@pytest.mark.parametrize("variant", sorted(GRID_VARIANTS))
def test_registered_kljun_matches_reference_across_grid_specs(variant):
    """Every grid-resolution branch stays bitwise identical to the reference."""
    grid = GRID_VARIANTS[variant]
    met = CASES["unstable-z0"]
    fp = get_model("kljun2015")(**met, **grid)
    out = reference.FFP_climatology(
        zm=met["zm"], z0=met.get("z0"), umean=met.get("umean"),
        ustar=met["ustar"], h=met["pblh"], ol=met["mo_length"],
        sigmav=met["v_sigma"], wind_dir=met["wind_dir"],
        rs=None, verbosity=0, fig=False, **grid)
    expected = np.asarray(out["fclim_2d"])
    assert fp.f.dtype == expected.dtype
    assert fp.f.shape == expected.shape
    assert np.array_equal(fp.f, expected)
    assert np.array_equal(fp.x, np.asarray(out["x_2d"])[0, :])
    assert np.array_equal(fp.y, np.asarray(out["y_2d"])[:, 0])


def test_port_matches_reference_direct_call():
    """Belt and braces: the raw port function, bypassing the adapter."""
    met = CASES["composite"]
    ported = port.calc_ffp_climatology(v_sigma=met["v_sigma"], verbosity=0,
                                       **{k: v for k, v in met.items()
                                          if k != "v_sigma"}, **GRID)
    expected, n = _kljun2015_reference(**met)
    assert ported.n == n
    assert np.array_equal(np.asarray(ported.fclim_2d), expected)


def test_crosswind_integrated_peak_matches_analytic_scaling():
    """Pin x_ci_max against the analytic xstar_max = -c/b + d (Kljun 2015)."""
    out = port.calc_footprint_1d(
        zm=20.0, z0=0.1, pblh=1000.0, mo_length=-100.0, v_sigma=0.5,
        ustar=0.5)
    assert out["flag_err"] == 0
    assert out["x_ci_max"] == pytest.approx(84.94, abs=0.05)
    # The crosswind-integrated footprint integrates to ~0.95 up to xstar = 30.
    integral = np.trapezoid(out["f_ci"], out["x_ci"])
    assert integral == pytest.approx(0.952, abs=0.01)
