"""Tests for fluxprint.footprint (Footprint + FootprintSeries).

Construction/geometry/aggregation need only numpy. The georeference tests run
against an identity pyproj stand-in (the translation arithmetic is validated;
projection correctness is delegated to pyproj). NetCDF/TIFF tests use their
optional backends where available.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from fluxprint.footprint import Footprint, FootprintSeries


def _toy(nx=5, ny=5, dx=10.0, **kw):
    f = np.zeros((ny, nx))
    f[ny // 2, nx // 2] = 1.0
    kw.setdefault("tower", (1000.0, 2000.0))
    kw.setdefault("tower_crs", "EPSG:3035")
    return Footprint.from_grid(f, dx=dx, **kw)


# --------------------------------------------------------------------------- #
# Footprint: construction / validation                                        #
# --------------------------------------------------------------------------- #
def test_from_grid_is_centred_on_tower():
    fp = _toy()
    assert fp.x.tolist() == [-20, -10, 0, 10, 20]
    assert fp.y.tolist() == [-20, -10, 0, 10, 20]
    assert fp.dx == 10.0 and fp.dy == 10.0
    assert fp.is_georeferenced is False


def test_shape_coordinate_mismatch_raises():
    with pytest.raises(ValueError, match="coordinates imply"):
        Footprint(f=np.zeros((3, 4)), x=np.arange(3), y=np.arange(3))


def test_non_2d_field_raises():
    with pytest.raises(ValueError, match="2-D"):
        Footprint.from_grid(np.zeros((2, 3, 4)), dx=10.0)


def test_irregular_grid_raises():
    with pytest.raises(ValueError, match="regularly spaced"):
        Footprint(f=np.zeros((1, 3)), x=np.array([0.0, 1.0, 5.0]), y=np.array([0.0]))


def test_frame_invariant_georeferenced_forbids_relative_time():
    with pytest.raises(ValueError, match="relative"):
        Footprint.from_grid(np.zeros((2, 2)), dx=10.0, crs="EPSG:3035", time=13.0)


def test_local_frame_allows_relative_time_label():
    fp = Footprint.from_grid(np.zeros((2, 2)), dx=10.0, time=13.0)  # crs is None
    assert fp.time == 13.0


# --------------------------------------------------------------------------- #
# Footprint: geometry / analysis                                              #
# --------------------------------------------------------------------------- #
def test_total_is_sum_times_cell_area():
    assert _toy(dx=10.0).total() == pytest.approx(100.0)


def test_peak_xy_at_origin():
    assert _toy().peak_xy() == (0.0, 0.0)


def test_normalized_integrates_to_one():
    # The *integral* (sum * dx * dy) is one, honouring the documented contract;
    # the plain sum is 1/(dx*dy).
    fp = Footprint.from_grid(np.array([[1.0, 3.0], [2.0, 2.0]]), dx=2.0)
    assert fp.normalized().total() == pytest.approx(1.0)
    assert np.nansum(fp.normalized().f) == pytest.approx(1.0 / 4.0)


def test_is_climatology_flag():
    assert _toy().is_climatology is True
    assert _toy()._replace(time=datetime(2024, 4, 24)).is_climatology is False


# --------------------------------------------------------------------------- #
# Footprint: georeference (identity pyproj stand-in)                          #
# --------------------------------------------------------------------------- #
def test_georeference_translates_grid_to_tower_position():
    pytest.importorskip("pyproj")
    fp = _toy(tower=(1000.0, 2000.0), tower_crs="EPSG:3035")
    g = fp.georeference("EPSG:3035")
    assert g.is_georeferenced is True
    assert g.crs == "EPSG:3035"
    assert g.x.tolist() == [980, 990, 1000, 1010, 1020]   # local + tower_x
    assert g.y.tolist() == [1980, 1990, 2000, 2010, 2020]  # local + tower_y


def test_georeference_twice_raises():
    pytest.importorskip("pyproj")
    g = _toy().georeference("EPSG:3035")
    with pytest.raises(ValueError, match="already georeferenced"):
        g.georeference("EPSG:3035")


def test_georeference_without_tower_raises():
    pytest.importorskip("pyproj")
    fp = Footprint.from_grid(np.zeros((2, 2)), dx=10.0)  # no tower
    with pytest.raises(ValueError, match="tower"):
        fp.georeference("EPSG:3035")


def test_to_lonlat_requires_georeferenced():
    pytest.importorskip("pyproj")
    with pytest.raises(ValueError, match="georeferenced"):
        _toy().to_lonlat()


def test_to_lonlat_returns_2d_grids():
    pytest.importorskip("pyproj")
    lon, lat = _toy().georeference("EPSG:3035").to_lonlat()
    assert lon.shape == (5, 5) and lat.shape == (5, 5)


# --------------------------------------------------------------------------- #
# Footprint: NetCDF / TIFF                                                     #
# --------------------------------------------------------------------------- #
def test_xarray_roundtrip_preserves_fields_and_metadata():
    pytest.importorskip("xarray")
    fp = _toy()._replace(time=datetime(2024, 4, 24), n=48)
    fp.attrs["site"] = "demo"
    out = Footprint.from_xarray(fp.to_xarray())
    assert np.array_equal(out.f, fp.f)
    assert np.array_equal(out.x, fp.x) and np.array_equal(out.y, fp.y)
    assert out.tower == (1000.0, 2000.0) and out.tower_crs == "EPSG:3035"
    assert out.n == 48
    assert out.attrs["site"] == "demo"
    assert out.time == datetime(2024, 4, 24)


def _netcdf_engine():
    for c in ("netcdf4", "h5netcdf", "scipy"):
        try:
            __import__(c)
            return c
        except ImportError:
            continue
    return None


def test_netcdf_disk_roundtrip(tmp_path):
    pytest.importorskip("xarray")
    engine = _netcdf_engine()
    if engine is None:
        pytest.skip("no NetCDF backend available")
    fp = _toy()._replace(n=10)
    path = tmp_path / "fp.nc"
    fp.to_netcdf(str(path), engine=engine)
    out = Footprint.from_netcdf(str(path), engine=engine)
    assert np.allclose(out.f, fp.f)
    assert out.n == 10


def test_to_tiff_requires_georeferenced():
    pytest.importorskip("rasterio")
    with pytest.raises(ValueError, match="georeferenced"):
        _toy().to_tiff("/tmp/should_not_write.tif")


def test_tiff_roundtrip_north_up_and_grid(tmp_path):
    pytest.importorskip("rasterio")
    import rasterio
    f = np.array([[1.0, 2.0], [3.0, 4.0]])  # row 0 = south
    fp = Footprint.from_grid(f, dx=10.0, tower=(0.0, 0.0),
                             tower_crs="EPSG:3035").georeference("EPSG:3035")
    path = tmp_path / "fp.tif"
    fp.to_tiff(str(path))
    with rasterio.open(str(path)) as src:
        raw = src.read(1)
    assert np.allclose(raw[0], [3.0, 4.0])  # written north-up: north row first
    out = Footprint.from_tiff(str(path))
    assert np.allclose(out.f, f)            # round-trips back to south-first
    assert out.crs == "EPSG:3035"
    assert np.allclose(out.x, fp.x) and np.allclose(out.y, fp.y)


# --------------------------------------------------------------------------- #
# FootprintSeries                                                             #
# --------------------------------------------------------------------------- #
def _series(n=3):
    fps = []
    for i in range(n):
        f = np.full((2, 2), float(i))
        fps.append(Footprint.from_grid(f, dx=10.0, time=datetime(2024, 4, 24, i),
                                       tower=(0.0, 0.0), tower_crs="EPSG:3035", n=i + 1))
    return FootprintSeries(fps)


def test_series_requires_nonempty():
    with pytest.raises(ValueError, match="at least one"):
        FootprintSeries([])


def test_series_rejects_mismatched_grid():
    a = Footprint.from_grid(np.zeros((2, 2)), dx=10.0)
    b = Footprint.from_grid(np.zeros((2, 3)), dx=10.0)
    with pytest.raises(ValueError, match="same grid"):
        FootprintSeries([a, b])


def test_series_rejects_mismatched_crs():
    # Same grid (tower at origin -> identity translation), differing only in crs.
    a = Footprint.from_grid(np.zeros((2, 2)), dx=10.0, tower=(0.0, 0.0),
                            tower_crs="EPSG:3035").georeference("EPSG:3035")
    b = Footprint.from_grid(np.zeros((2, 2)), dx=10.0)  # local
    with pytest.raises(ValueError, match="same crs"):
        FootprintSeries([a, b])


def test_series_basics():
    s = _series(3)
    assert s.nt == 3 and len(s) == 3
    assert s.stack().shape == (3, 2, 2)
    assert [t.hour for t in s.times] == [0, 1, 2]
    assert s[1].time == datetime(2024, 4, 24, 1)


def test_series_aggregate_means_and_sums_counts():
    s = _series(3)                       # fields 0, 1, 2 -> mean 1.0
    clim = s.aggregate(smooth=False)
    assert np.allclose(clim.f, 1.0)
    assert clim.n == 1 + 2 + 3           # counts summed
    assert clim.is_climatology is True


def test_series_georeference_all_members():
    pytest.importorskip("pyproj")
    s = _series(2)
    g = s.georeference("EPSG:3035")
    assert g.is_georeferenced is True
    assert all(fp.crs == "EPSG:3035" for fp in g)
    assert np.allclose(g.x, s.x)         # tower at origin -> identity translation


def test_series_georeference_requires_timestamps():
    pytest.importorskip("pyproj")
    fps = [Footprint.from_grid(np.zeros((2, 2)), dx=10.0, time=float(i),
                               tower=(0.0, 0.0), tower_crs="EPSG:3035")
           for i in range(2)]
    with pytest.raises(ValueError, match="datetime"):
        FootprintSeries(fps).georeference("EPSG:3035")


def test_series_xarray_roundtrip():
    pytest.importorskip("xarray")
    s = _series(3)
    ds = s.to_xarray()
    assert ds["footprint"].dims == ("time", "y", "x")
    out = FootprintSeries.from_xarray(ds)
    assert out.nt == 3
    assert np.allclose(out.stack(), s.stack())
    assert [t for t in out.times] == [t for t in s.times]
    assert [fp.n for fp in out] == [1, 2, 3]


# --------------------------------------------------------------------------- #
# CRS defaults / NetCDF packing (small-task additions)                        #
# --------------------------------------------------------------------------- #
def test_laea_crs_is_centred_on_point():
    from fluxprint.footprint import laea_crs
    s = laea_crs(48.8, 2.4)
    assert "+proj=laea" in s and "lat_0=48.8" in s and "lon_0=2.4" in s


def test_georeference_defaults_to_tower_laea():
    pytest.importorskip("pyproj")
    g = _toy().georeference()  # no target -> tower-centred LAEA
    assert g.is_georeferenced
    assert "laea" in g.crs


def test_to_netcdf_packing_roundtrips(tmp_path):
    pytest.importorskip("xarray")
    engine = _netcdf_engine()
    if engine is None:
        pytest.skip("no NetCDF backend available")
    fp = _toy()
    path = tmp_path / "packed.nc"
    fp.to_netcdf(str(path), decimals=6, engine=engine)
    out = Footprint.from_netcdf(str(path), engine=engine)
    assert np.allclose(out.f, fp.f, atol=1e-6)


# --------------------------------------------------------------------------- #
# NetCDF-safe attrs coercion / aggregate provenance                           #
# --------------------------------------------------------------------------- #
def test_coerce_attr_handles_numpy_bool_and_timestamps():
    # np.bool_ is an np.generic that netCDF writers reject; it must become int.
    assert Footprint._coerce_attr(np.bool_(True)) == 1
    assert not isinstance(Footprint._coerce_attr(np.bool_(True)), np.bool_)
    assert Footprint._coerce_attr([np.bool_(True), np.bool_(False)]) == [1, 0]
    assert Footprint._coerce_attr(datetime(2024, 4, 24)) == "2024-04-24T00:00:00"
    assert Footprint._coerce_attr(object()).startswith("<object")


def test_aggregate_carries_shared_attrs_and_drops_per_member_ones():
    fps = []
    for i in range(2):
        fp = Footprint.from_grid(np.ones((3, 3)), dx=10.0, n=1, time=float(i))
        fp.attrs.update({"model": "kljun2015",
                         "estimated_inputs": ["v_sigma"],
                         "group": f"g{i}"})
        fps.append(fp)
    clim = FootprintSeries(fps).aggregate(smooth=False)
    assert clim.attrs["model"] == "kljun2015"
    assert clim.attrs["estimated_inputs"] == ["v_sigma"]
    assert "group" not in clim.attrs          # differs per member -> dropped
