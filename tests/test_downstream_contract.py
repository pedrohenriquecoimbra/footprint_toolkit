"""Pins the public surface downstream integrations depend on.

fluxcom's FFPProvider adapter embeds FluxPrint as a compute kernel and pins
exactly this surface: ``get_model("kljun2015")`` with its keyword arguments
(including ``smooth_data``), ``fluxprint.micrometeorology.filler``, and the
``Footprint.f``/``.x``/``.y`` attributes. Renaming any of it is a breaking
change for fluxcom — this file is the tripwire. ``smooth_data`` in particular
stays a supported spelling throughout 0.x even if a generic alias is added.
"""
from __future__ import annotations

import inspect

import numpy as np


#: Keyword arguments the fluxcom adapter passes to get_model("kljun2015").
PINNED_MODEL_KWARGS = (
    "zm", "ustar", "pblh", "mo_length", "v_sigma", "wind_dir",
    "z0", "umean", "domain", "dx", "dy", "nx", "ny",
    "smooth_data", "verbosity",
)


def test_get_model_kljun2015_accepts_pinned_kwargs():
    from fluxprint.model import get_model

    model = get_model("kljun2015")
    params = inspect.signature(model).parameters
    missing = [k for k in PINNED_MODEL_KWARGS if k not in params]
    assert not missing, (
        f"get_model('kljun2015') lost downstream-pinned kwarg(s) {missing}; "
        "fluxcom's FFPProvider passes these by name.")


def test_micrometeorology_filler_callable_as_pinned():
    from fluxprint.micrometeorology import filler

    params = list(inspect.signature(filler).parameters.values())
    # Called as filler(data, key, fill_all=...): two leading positionals
    # plus a fill_all keyword.
    assert len(params) >= 2
    assert all(p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
               for p in params[:2])
    assert "fill_all" in inspect.signature(filler).parameters
    # And it works: a physically grounded estimate with fill_all=False.
    value = filler({"ustar": [0.4], "umean": [3.0], "zm": [20.0],
                    "mo_length": [-100.0]}, "v_sigma", fill_all=False)
    assert value is not None


def test_model_returns_footprint_with_f_x_y():
    from fluxprint.model import get_model

    fp = get_model("kljun2015")(
        zm=20.0, ustar=0.5, pblh=1000.0, mo_length=-100.0, v_sigma=0.5,
        wind_dir=30.0, z0=0.1, domain=[-100.0, 100.0, -100.0, 100.0],
        dx=10.0, smooth_data=0, verbosity=0)
    assert isinstance(fp.f, np.ndarray) and fp.f.ndim == 2
    assert isinstance(fp.x, np.ndarray) and fp.x.ndim == 1
    assert isinstance(fp.y, np.ndarray) and fp.y.ndim == 1
    assert fp.f.shape == (fp.y.size, fp.x.size)
