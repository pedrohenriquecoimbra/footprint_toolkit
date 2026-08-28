"""Tests for fluxprint.mapping.map_footprints: the xarray/dask-native path.

The mapped path must reproduce single-record model calls exactly (same
kernel, same validation), keep input dims, resolve aliases, honour on_error,
and stay lazy over dask-backed inputs.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("rasterio")  # the package import pulls the geo stack
xr = pytest.importorskip("xarray")

from fluxprint import map_footprints  # noqa: E402
from fluxprint.model import get_model  # noqa: E402

GRID = dict(domain=[-200.0, 200.0, -200.0, 200.0], dx=10.0)
MET = dict(zm=20.0, z0=0.1, ustar=0.5, pblh=1000.0, mo_length=-100.0,
           v_sigma=0.5)


def _dataset(wind_dirs=(30.0, 180.0, 270.0)):
    time = np.arange(len(wind_dirs))
    return xr.Dataset({
        "wind_dir": ("time", np.asarray(wind_dirs, dtype=float)),
        **{k: ("time", np.full(len(wind_dirs), v)) for k, v in MET.items()},
    }, coords={"time": time})


def test_mapped_records_match_single_model_calls_exactly():
    ds = _dataset()
    result = map_footprints(ds, **GRID)
    assert result.dims == ("time", "y", "x")
    kljun = get_model("kljun2015")
    for i, wd in enumerate(ds["wind_dir"].values):
        fp = kljun(**MET, wind_dir=float(wd), **GRID, smooth_data=0,
                   verbosity=0)
        assert np.array_equal(result.isel(time=i).values, fp.f), (
            f"mapped record {i} differs from the single-record model call")
        assert np.array_equal(result["x"].values, fp.x)
        assert np.array_equal(result["y"].values, fp.y)


def test_scalars_broadcast_against_arrays():
    ds = xr.Dataset({"wind_dir": ("time", [30.0, 210.0])})
    result = map_footprints(ds, **MET, **GRID)
    assert result.dims == ("time", "y", "x")
    assert np.isfinite(result.values).all()


def test_nd_input_dims_are_preserved():
    wind = xr.DataArray(np.full((2, 3), 180.0), dims=("site", "hour"))
    ustar = xr.DataArray([0.5, 0.4], dims=("site",))
    result = map_footprints(wind_dir=wind, ustar=ustar,
                            **{k: v for k, v in MET.items() if k != "ustar"},
                            **GRID)
    assert result.dims == ("site", "hour", "y", "x")
    assert result.shape[:2] == (2, 3)


def test_aliases_resolve_in_datasets():
    ds = _dataset().rename({"wind_dir": "WD", "ustar": "USTAR",
                            "pblh": "blh", "mo_length": "OL",
                            "v_sigma": "sigma_v"})
    named = map_footprints(ds, **GRID)
    canonical = map_footprints(_dataset(), **GRID)
    assert np.array_equal(named.values, canonical.values)


def test_on_error_nan_gives_nan_planes_only_for_bad_records():
    ds = _dataset()
    ustar = np.full(3, MET["ustar"])
    ustar[1] = 0.05  # fails validation (ustar <= 0.1)
    ds["ustar"] = ("time", ustar)
    result = map_footprints(ds, **GRID)
    assert np.isnan(result.isel(time=1).values).all()
    assert np.isfinite(result.isel(time=0).values).all()
    assert np.isfinite(result.isel(time=2).values).all()


def test_on_error_raise_aborts():
    ds = _dataset()
    ds["ustar"] = ("time", np.array([0.5, 0.05, 0.5]))
    with pytest.raises(Exception, match="invalid input"):
        map_footprints(ds, on_error="raise", **GRID)


def test_smooth_flag_applies_the_generic_kernel():
    from fluxprint.footprint import smooth_field

    ds = _dataset(wind_dirs=(30.0,))
    raw = map_footprints(ds, **GRID)
    smoothed = map_footprints(ds, smooth=1, **GRID)
    assert np.array_equal(smoothed.isel(time=0).values,
                          smooth_field(raw.isel(time=0).values))


def test_model_options_are_forwarded():
    # rslayer=1 permits a record inside the roughness sublayer that the
    # default validation rejects.
    low = dict(MET, zm=1.0)  # zm <= 12.5 * z0
    ds = _dataset(wind_dirs=(30.0,))
    for k, v in low.items():
        ds[k] = ("time", np.array([v]))
    rejected = map_footprints(ds, **GRID)
    allowed = map_footprints(ds, rslayer=1, **GRID)
    assert np.isnan(rejected.values).all()
    assert np.isfinite(allowed.values).all()


def test_provenance_attrs_and_missing_inputs():
    result = map_footprints(_dataset(), **GRID)
    assert result.attrs["model"] == "kljun2015"
    assert result.attrs["units"] == "m-2"
    assert "fluxprint_version" in result.attrs
    with pytest.raises(ValueError, match="Missing required met variable"):
        map_footprints(xr.Dataset({"wind_dir": ("time", [30.0])}), **GRID)
    with pytest.raises(TypeError, match="Unsupported `data` container"):
        map_footprints([1, 2, 3], **GRID)


def test_dask_backed_input_stays_lazy_and_matches_eager():
    pytest.importorskip("dask")
    ds = _dataset(wind_dirs=tuple(np.linspace(0, 350, 8)))
    lazy = map_footprints(ds.chunk({"time": 3}), **GRID)
    assert lazy.chunks is not None  # still lazy, nothing computed yet
    eager = map_footprints(ds, **GRID)
    assert np.array_equal(lazy.compute().values, eager.values)
