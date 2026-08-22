"""Golden regression tests pinning the Kljun port to the vendored reference.

The package vendors the original FFP code (fluxprint/model/Kljun_et_al_2015_original)
precisely so the rewritten port can be validated against it. These tests keep
the two numerically identical: any future edit to the port that changes the
field will fail here.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("rasterio")  # the package import pulls the geo stack

from fluxprint.model import Kljun_et_al_2015 as port  # noqa: E402
from fluxprint.model.Kljun_et_al_2015_original import (  # noqa: E402
    calc_footprint_FFP_climatology as reference)

GRID = dict(domain=[-300, 300, -300, 300], dx=10.0, dy=10.0)

#: (label, ol) — one case per stability regime. zm/ol stays within validity.
REGIMES = [
    ("unstable", -100.0),
    ("neutral", 1.0e6),     # |ol| > 5000 -> the model's neutral branch
    ("stable", 200.0),
]


def _run_both(*, z0=None, umean=None, ol=-100.0):
    met = dict(zm=[20.0], ustar=[0.5], wind_dir=[30.0])
    ported = port.calc_ffp_climatology(
        z0=[z0], umean=[umean], pblh=[1000.0], mo_length=[ol],
        v_sigma=[0.5], verbosity=0, **met, **GRID)
    ref = reference.FFP_climatology(
        z0=[z0], umean=[umean], h=[1000.0], ol=[ol], sigmav=[0.5],
        rs=None, verbosity=0, fig=False, **met, **GRID)
    return ported, ref


@pytest.mark.parametrize("label,ol", REGIMES)
def test_port_matches_reference_z0_path(label, ol):
    ported, ref = _run_both(z0=0.1, ol=ol)
    assert ported.n == ref["n"] == 1, label
    assert np.allclose(ported.fclim_2d, ref["fclim_2d"], rtol=1e-12, atol=0), label


def test_port_matches_reference_umean_path():
    ported, ref = _run_both(umean=3.0)
    assert ported.n == ref["n"] == 1
    assert np.allclose(ported.fclim_2d, ref["fclim_2d"], rtol=1e-12, atol=0)


def test_port_matches_reference_multirecord_composite():
    met = dict(zm=[20.0] * 3, z0=[0.1] * 3, ustar=[0.5, 0.4, 0.3],
               wind_dir=[30.0, 180.0, 270.0])
    ported = port.calc_ffp_climatology(
        pblh=[1000.0] * 3, mo_length=[-100.0, -50.0, 200.0],
        v_sigma=[0.5] * 3, verbosity=0, **met, **GRID)
    ref = reference.FFP_climatology(
        h=[1000.0] * 3, ol=[-100.0, -50.0, 200.0], sigmav=[0.5] * 3,
        rs=None, verbosity=0, fig=False, **met, **GRID)
    assert ported.n == ref["n"] == 3
    assert np.allclose(ported.fclim_2d, ref["fclim_2d"], rtol=1e-12, atol=0)


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
