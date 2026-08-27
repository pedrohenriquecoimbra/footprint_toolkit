"""Integration tests for the rewritten ``core.calculate_footprint`` / ``wrapper``.

These run the real Kljun model (numpy + scipy) through the registry and assert
the new ``FootprintSeries`` return type, grouping, and model resolution. The
package import pulls the geo stack, so the suite is skipped without it.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("rasterio")  # the package imports rasterio/xarray/pyproj eagerly

import pandas as pd  # noqa: E402

from fluxprint import core  # noqa: E402
from fluxprint.footprint import Footprint, FootprintSeries  # noqa: E402

SMALL = dict(dx=20.0, domain=[-200, 200, -200, 200])  # small grid -> fast


def _frame(groups=("A", "A", "B", "B")):
    n = len(groups)
    return pd.DataFrame({
        "plot": list(groups),
        "zm": [2.0] * n,
        "umean": [3.0] * n,
        "ustar": [0.5, 0.4, 0.5, 0.4][:n],
        "pblh": [1000.0] * n,
        "mo_length": [-50.0] * n,
        "v_sigma": [0.5] * n,
        "wind_dir": [180.0, 200.0, 180.0, 200.0][:n],
    })


# --------------------------------------------------------------------------- #
# Return type and grouping                                                    #
# --------------------------------------------------------------------------- #
def test_by_none_returns_single_climatology_series():
    series = core.calculate_footprint(_frame(), model="kljun2015", **SMALL)
    assert isinstance(series, FootprintSeries)
    assert series.nt == 1
    assert series[0].n == 4              # all rows composited into one footprint
    assert series[0].is_climatology      # no single timestamp
    assert series[0].is_georeferenced is False


def test_grouping_yields_one_footprint_per_group():
    series = core.calculate_footprint(_frame(), by="plot", model="kljun2015", **SMALL)
    assert series.nt == 2
    assert sorted(fp.attrs["group"] for fp in series) == ["A", "B"]
    assert all(fp.n == 2 for fp in series)            # 2 records per group
    assert series.aggregate().is_climatology          # collapses to a climatology


def test_datetime_grouping_sets_real_timestamps():
    from datetime import datetime
    df = _frame()
    df["t"] = [datetime(2024, 4, 24, h) for h in (0, 0, 1, 1)]
    series = core.calculate_footprint(df, by="t", model="kljun2015", **SMALL)
    assert series.nt == 2
    assert all(isinstance(fp.time, datetime) for fp in series)


def test_tower_metadata_enables_georeferencing():
    pytest.importorskip("pyproj")
    series = core.calculate_footprint(
        _frame(), model="kljun2015",
        tower=(4321000.0, 3210000.0), tower_crs="EPSG:3035", **SMALL)
    geo = series[0].georeference("EPSG:3035")
    assert geo.is_georeferenced
    assert np.isclose(geo.x.mean(), 4321000.0, atol=1.0)


# --------------------------------------------------------------------------- #
# Model resolution                                                            #
# --------------------------------------------------------------------------- #
def test_model_resolves_by_name_callable_and_module():
    df = _frame(("A", "A"))
    from fluxprint.model.Kljun_et_al_2015 import calc      # the callable
    from fluxprint.model import kljun2015 as kljun_module  # a module exposing calc

    by_name = core.calculate_footprint(df, model="kljun2015", **SMALL)
    by_callable = core.calculate_footprint(df, model=calc, **SMALL)
    by_module = core.calculate_footprint(df, model=kljun_module, **SMALL)

    assert by_name.nt == by_callable.nt == by_module.nt == 1
    assert np.allclose(by_name[0].f, by_callable[0].f)
    assert np.allclose(by_name[0].f, by_module[0].f)


def test_unknown_model_name_raises():
    with pytest.raises(KeyError, match="kljun2015"):
        core.calculate_footprint(_frame(("A", "A")), model="nope", **SMALL)


def test_bad_model_type_raises():
    with pytest.raises(TypeError, match="model"):
        core.calculate_footprint(_frame(("A", "A")), model=12345, **SMALL)


# --------------------------------------------------------------------------- #
# wrapper                                                                     #
# --------------------------------------------------------------------------- #
def test_wrapper_aggregates_to_footprint_by_default():
    result = core.wrapper(_frame(), model="kljun2015", **SMALL)
    assert isinstance(result, Footprint)
    assert result.is_climatology
    assert result.attrs["model"] == "kljun2015"
    assert "model_used" not in result.attrs   # the legacy duplicate key is gone


def test_wrapper_resolves_callable_model_to_name():
    # A dummy model that does NOT stamp attrs["model"] itself, so this test
    # actually exercises the wrapper's callable->name resolution.
    def dummy(*, dx, time=None, tower=None, tower_crs=None, **kw):
        return Footprint.from_grid(np.ones((5, 5)), dx=dx, time=time, n=1)

    result = core.wrapper(_frame(), model=dummy, aggregate=False, **SMALL)
    assert all(fp.attrs["model"] == "dummy" for fp in result)


def test_wrapper_carries_provenance_onto_climatology():
    df = _frame().drop(columns=["v_sigma"])   # v_sigma -> estimated from ustar
    result = core.wrapper(df, model="kljun2015", **SMALL)
    assert "v_sigma" in result.attrs["estimated_inputs"]


def test_wrapper_returns_series_when_not_aggregated():
    result = core.wrapper(_frame(), aggregate=False, model="kljun2015", **SMALL)
    assert isinstance(result, FootprintSeries)


def test_wrapper_tiff_requires_georeferenced():
    with pytest.raises(ValueError, match="[Gg]eoreference"):
        core.wrapper(_frame(), out_as="tif", dst="/tmp/should_not_write.tif",
                     model="kljun2015", **SMALL)


def _poisoned_frame():
    """Group B's records all fail validation (ustar <= 0.1) -> n == 0."""
    df = _frame()
    df.loc[df["plot"] == "B", "ustar"] = 0.05
    return df


def test_on_error_default_skip_drops_failed_group():
    series = core.calculate_footprint(_poisoned_frame(), by="plot",
                                      model="kljun2015", **SMALL)
    assert series.nt == 1
    assert series[0].attrs["group"] == "A"


def test_on_error_nan_keeps_group_slots():
    series = core.calculate_footprint(_poisoned_frame(), by="plot",
                                      model="kljun2015", on_error="nan",
                                      **SMALL)
    assert series.nt == 2
    good = next(fp for fp in series if fp.attrs["group"] == "A")
    bad = next(fp for fp in series if fp.attrs["group"] == "B")
    assert good.n == 2 and np.isfinite(good.f).all()
    assert bad.n == 0
    assert np.isnan(bad.f).all()
    assert bad.attrs["error"] == "no valid records in group"
    assert bad.f.shape == good.f.shape  # same grid: the series stays regular


def test_on_error_raise_aborts_on_invalid_group():
    with pytest.raises(ValueError, match="no valid records"):
        core.calculate_footprint(_poisoned_frame(), by="plot",
                                 model="kljun2015", on_error="raise", **SMALL)


def test_on_error_nan_on_model_exception():
    from fluxprint.model import get_model
    kljun = get_model("kljun2015")
    calls = {"n": 0}

    def flaky_model(**kw):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("boom")
        return kljun(**kw)

    series = core.calculate_footprint(_frame(), by="plot", model=flaky_model,
                                      on_error="nan", **SMALL)
    assert series.nt == 2
    bad = [fp for fp in series if "error" in fp.attrs]
    assert len(bad) == 1
    assert "boom" in bad[0].attrs["error"]
    assert np.isnan(bad[0].f).all()


def test_on_error_rejects_unknown_policy():
    with pytest.raises(ValueError, match="on_error"):
        core.calculate_footprint(_frame(), model="kljun2015",
                                 on_error="ignore", **SMALL)


def test_empty_footprint_has_grid_and_nan_field():
    t = core.empty_footprint(model="kljun2015", dx=20, domain=[-200, 200, -200, 200])
    assert isinstance(t, Footprint)
    nx = int(400 / 20)
    assert t.f.shape == (nx + 1, nx + 1)
    assert np.isnan(t.f).all()
    assert not t.is_georeferenced


def test_empty_footprint_grid_identical_to_a_model_run():
    """The grid-hook path must reproduce the model's own grid exactly."""
    from fluxprint.model import get_model

    t = core.empty_footprint(model="kljun2015", dx=20,
                             domain=[-200, 200, -200, 200])
    fp = get_model("kljun2015")(
        zm=2.0, umean=2.0, ustar=0.3, pblh=1000.0, mo_length=-100.0,
        v_sigma=0.5, wind_dir=0.0, dx=20, domain=[-200, 200, -200, 200],
        verbosity=0)
    assert np.array_equal(t.x, fp.x)
    assert np.array_equal(t.y, fp.y)
    assert t.f.shape == fp.f.shape
