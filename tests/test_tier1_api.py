"""Tests for the Tier 1 API consolidation.

Covers the new Footprint analysis/export methods (contours, plot,
to_shapefile), the deprecation shims over the legacy surface, the fixed
calc_footprint_1d validation, and the model-registry cleanup.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("rasterio")  # the package import pulls the geo stack

from fluxprint import core, io  # noqa: E402
from fluxprint.exceptions import InputValidationError  # noqa: E402
from fluxprint.footprint import Footprint, FootprintSeries  # noqa: E402
from fluxprint.model import available_models  # noqa: E402


def _gaussian(sigma=50.0, extent=300.0, dx=5.0, **kwargs):
    """Unit-integral 2-D Gaussian footprint (analytically known contours)."""
    x = np.arange(-extent, extent + dx, dx)
    y = np.arange(-extent, extent + dx, dx)
    xx, yy = np.meshgrid(x, y)
    f = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2)) / (2 * np.pi * sigma ** 2)
    return Footprint(f=f, x=x, y=y, **kwargs)


# --------------------------------------------------------------------------- #
# Footprint.contours                                                          #
# --------------------------------------------------------------------------- #
def test_contours_match_analytic_gaussian():
    sigma = 50.0
    fp = _gaussian(sigma=sigma)
    (c,) = fp.contours(0.5)

    assert c["r"] == 0.5
    assert c["closed"] is True
    assert c["fraction"] == pytest.approx(0.5, abs=0.02)
    # The 50% region of a symmetric Gaussian is a circle of radius
    # sigma * sqrt(2 ln 2) ~ 58.87 m.
    vertices = np.vstack(c["vertices"])
    radii = np.hypot(vertices[:, 0], vertices[:, 1])
    assert radii.mean() == pytest.approx(sigma * np.sqrt(2 * np.log(2)), rel=0.03)


def test_contours_accepts_percentages_and_sorts():
    fp = _gaussian()
    out = fp.contours([80, 0.5])
    assert [c["r"] for c in out] == [0.5, 0.8]
    assert out[0]["level"] > out[1]["level"]   # smaller area -> higher level


def test_contours_warns_on_unreachable_fraction():
    fp = _gaussian(sigma=200.0, extent=100.0)  # domain captures only ~15%
    with pytest.warns(UserWarning, match="unreachable"):
        (c,) = fp.contours(0.8)
    assert np.isnan(c["level"])
    assert c["vertices"] == []


def test_contours_rejects_fractions_beyond_90_percent():
    with pytest.raises(ValueError, match="0.9"):
        _gaussian().contours(0.95)


def test_contours_on_model_output():
    from fluxprint.model.Kljun_et_al_2015 import calc

    fp = calc(zm=10.0, z0=0.1, ustar=0.5, pblh=1000.0, mo_length=-100.0,
              v_sigma=0.5, wind_dir=180.0, dx=10.0,
              domain=[-300, 300, -300, 300])
    (c,) = fp.contours(0.5)
    assert c["closed"] is True
    # Recompute the enclosed integral independently from the field, so a
    # mis-scaled level computation cannot satisfy this via its own numbers.
    enclosed = fp.f[fp.f >= c["level"]].sum() * fp.dx * fp.dy
    assert enclosed == pytest.approx(0.5, abs=0.03)


def test_contours_closed_detection_is_exact_at_georeferenced_coords(tmp_path):
    """A relative tolerance would mark open contours 'closed' at ~1e6 m."""
    fiona = pytest.importorskip("fiona")
    # sigma=60 puts the 90% radius (~129 m) outside the 125 m half-domain,
    # so the 90% isopleth is clipped by the edge and must be open.
    fp = _gaussian(sigma=60.0, extent=125.0, dx=5.0)
    geo = Footprint(f=fp.f, x=fp.x + 4.0e6, y=fp.y + 2.8e6, crs="EPSG:3035")

    (c,) = geo.contours(0.9)
    assert c["closed"] is False
    path = tmp_path / "open.shp"
    with pytest.warns(UserWarning, match="open"):
        geo.to_shapefile(str(path), rs=[0.9])
    with fiona.open(str(path)) as src:
        assert len(list(src)) == 0           # nothing silently bridged


# --------------------------------------------------------------------------- #
# Footprint.plot                                                              #
# --------------------------------------------------------------------------- #
def test_plot_returns_axes_without_showing():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fp = _gaussian()
    ax = fp.plot(rs=[0.5])
    assert ax.figure is not None
    assert len(ax.collections) >= 1          # the pcolormesh
    plt.close(ax.figure)


def test_plot_colorbar_only_on_new_figures_by_default():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fp = _gaussian(dx=20.0, extent=100.0)
    fig, ax = plt.subplots()
    fp.plot(ax=ax)
    assert len(fig.axes) == 1                # no colorbar stolen from caller
    fp.plot(ax=ax, colorbar=True)
    assert len(fig.axes) == 2                # explicit request honoured
    plt.close(fig)

    own = fp.plot()
    assert len(own.figure.axes) == 2         # new figure gets one by default
    plt.close(own.figure)


def test_plot_transforms_tower_marker_into_grid_crs():
    matplotlib = pytest.importorskip("matplotlib")
    pytest.importorskip("pyproj")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fp = _gaussian(dx=20.0, extent=100.0, tower=(5.0, 48.0),
                   tower_crs="EPSG:4326")
    geo = fp.georeference()                  # tower-centred LAEA
    ax = geo.plot(colorbar=False)
    marker = ax.lines[-1].get_xydata()[0]
    # In a tower-centred frame the marker must sit at the origin, not at the
    # raw lon/lat (5, 48) the tower is stored as.
    assert np.hypot(*marker) < 1.0
    plt.close(ax.figure)


# --------------------------------------------------------------------------- #
# Footprint.to_shapefile                                                      #
# --------------------------------------------------------------------------- #
def test_to_shapefile_roundtrip(tmp_path):
    fiona = pytest.importorskip("fiona")
    path = tmp_path / "contours.shp"
    _gaussian().to_shapefile(str(path), rs=[0.5, 0.8])

    with fiona.open(str(path)) as src:
        records = list(src)
    assert sorted(r["properties"]["r"] for r in records) == [0.5, 0.8]
    assert all(r["geometry"]["type"] == "Polygon" for r in records)


def test_to_shapefile_annulus_becomes_polygon_with_hole(tmp_path):
    """The hole of an annular source area must not export as a solid polygon."""
    fiona = pytest.importorskip("fiona")
    dx, extent, radius, width = 5.0, 300.0, 100.0, 20.0
    x = np.arange(-extent, extent + dx, dx)
    y = np.arange(-extent, extent + dx, dx)
    xx, yy = np.meshgrid(x, y)
    rr = np.hypot(xx, yy)
    f = np.exp(-((rr - radius) ** 2) / (2 * width ** 2))
    fp = Footprint(f=f / (f.sum() * dx * dx), x=x, y=y)

    path = tmp_path / "annulus.shp"
    fp.to_shapefile(str(path), rs=[0.5])
    with fiona.open(str(path)) as src:
        records = list(src)
    assert len(records) == 1
    rings = records[0]["geometry"]["coordinates"]
    assert len(rings) == 2                   # exterior + hole, one polygon
    outer = np.hypot(*np.asarray(rings[0]).T.astype(float)).mean()
    inner = np.hypot(*np.asarray(rings[1]).T.astype(float)).mean()
    assert outer > radius > inner


def test_to_shapefile_writes_crs_when_georeferenced(tmp_path):
    fiona = pytest.importorskip("fiona")
    pytest.importorskip("pyproj")
    fp = _gaussian()
    geo = Footprint(f=fp.f, x=fp.x + 4321000.0, y=fp.y + 3210000.0,
                    crs="EPSG:3035")
    path = tmp_path / "geo.shp"
    geo.to_shapefile(str(path), rs=[0.8])
    with fiona.open(str(path)) as src:
        assert src.crs_wkt


# --------------------------------------------------------------------------- #
# Deprecation shims over the legacy surface                                   #
# --------------------------------------------------------------------------- #
def test_get_contour_deprecated_but_delegates_for_footprints():
    with pytest.warns(DeprecationWarning, match="contours"):
        out = core.get_contour(_gaussian(), 5.0, 5.0, [0.5])
    assert out[0]["r"] == 0.5


def test_aggregate_footprints_removed():
    # Deprecated in 0.3.0, removed in 0.4.0 (one full cycle served).
    assert not hasattr(core, "aggregate_footprints")
    assert "aggregate_footprints" not in core.__all__


def test_legacy_write_netcdf_accepts_footprint(tmp_path):
    pytest.importorskip("xarray")
    fp = _gaussian(dx=20.0, extent=100.0)
    path = tmp_path / "legacy.nc"
    with pytest.warns(DeprecationWarning, match="write_to_file") as record:
        io.write_to_file(fp, str(path))
    # One warning, naming the function the user called - not the dispatchee.
    # Count only fluxprint's own deprecations: the netCDF write path can emit
    # unrelated DeprecationWarnings from numpy/xarray internals (e.g. numpy
    # 2.5's "setting the shape on an array" notice via xarray's netCDF4
    # backend), which are not this test's subject.
    ours = [w for w in record
            if w.category is DeprecationWarning
            and str(w.message).startswith("fluxprint.")]
    assert len(ours) == 1, [str(w.message) for w in ours]
    assert "write_to_file" in str(ours[0].message)
    out = Footprint.from_netcdf(str(path))
    assert np.allclose(out.f, fp.f, atol=1e-12)


def test_legacy_write_raster_accepts_georeferenced_footprint(tmp_path):
    fp = _gaussian(dx=20.0, extent=100.0)
    geo = Footprint(f=fp.f, x=fp.x + 4321000.0, y=fp.y + 3210000.0,
                    crs="EPSG:3035")
    path = tmp_path / "legacy.tif"
    with pytest.warns(DeprecationWarning, match="to_tiff"):
        io.write_to_raster(geo, str(path))
    assert path.exists()


def test_legacy_shapefile_writer_rejects_series_with_guidance():
    series = FootprintSeries([_gaussian(dx=20.0, extent=100.0)])
    with pytest.warns(DeprecationWarning):
        with pytest.raises(TypeError, match="aggregate"):
            io.write_to_shapefile(series, "unused.shp")


# --------------------------------------------------------------------------- #
# calc_footprint_1d: strict validation                                        #
# --------------------------------------------------------------------------- #
def test_calc_footprint_1d_runs_with_defaults():
    from fluxprint.model.Kljun_et_al_2015 import calc_footprint_1d

    out = calc_footprint_1d(zm=20.0, z0=0.1, pblh=1000.0, mo_length=-100.0,
                            v_sigma=0.5, ustar=0.5)
    assert out["flag_err"] == 0
    assert np.isfinite(out["f_ci"]).all()


def test_calc_footprint_1d_rejects_invalid_and_nan_inputs():
    from fluxprint.model.Kljun_et_al_2015 import calc_footprint_1d

    valid = dict(zm=20.0, z0=0.1, pblh=1000.0, mo_length=-100.0,
                 v_sigma=0.5, ustar=0.5)
    with pytest.raises(InputValidationError, match="ustar"):
        calc_footprint_1d(**{**valid, "ustar": 0.05})
    with pytest.raises(InputValidationError):
        calc_footprint_1d(**{**valid, "ustar": np.nan})
    with pytest.raises(InputValidationError, match="nx"):
        calc_footprint_1d(**valid, nx=100)
    with pytest.raises(InputValidationError):  # not ZeroDivisionError
        calc_footprint_1d(**{**valid, "mo_length": 0.0})


# --------------------------------------------------------------------------- #
# Model registry cleanup                                                      #
# --------------------------------------------------------------------------- #
def test_registry_exposes_exactly_the_working_models():
    assert available_models() == ["hsieh2000", "kljun2015"]


def test_kormann_meixner_stub_is_gone():
    with pytest.raises(ModuleNotFoundError):
        import fluxprint.model.Kormann_and_Meixner_2001  # noqa: F401


def test_broken_ffp_function_removed_from_port():
    from fluxprint.model import Kljun_et_al_2015 as kljun

    assert not hasattr(kljun, "FFP")
