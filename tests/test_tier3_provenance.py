"""Tests for provenance stamping, CF metadata, time policy and zm = z - d."""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

pytest.importorskip("rasterio")  # the package import pulls the geo stack

from fluxprint import core, micrometeorology  # noqa: E402
from fluxprint.footprint import Footprint  # noqa: E402
from fluxprint.model import get_model  # noqa: E402

MET = dict(zm=10.0, z0=0.1, ustar=0.5, pblh=1000.0, mo_length=-100.0,
           v_sigma=0.5, wind_dir=180.0)
GRID = dict(dx=20.0, domain=[-200, 200, -200, 200])


# --------------------------------------------------------------------------- #
# Provenance                                                                  #
# --------------------------------------------------------------------------- #
def test_model_output_carries_provenance():
    fp = get_model("kljun2015")(**MET, **GRID)
    assert fp.attrs["model"] == "kljun2015"
    assert fp.attrs["model_doi"] == "10.5194/gmd-8-3695-2015"
    assert "Kljun" in fp.attrs["model_citation"]
    assert fp.attrs["fluxprint_version"]
    assert fp.attrs["wind_profile_input"] == "z0"
    assert "fluxprint" in fp.attrs["history"]
    assert fp.attrs["smooth_data"] == 1


def test_wind_profile_input_reflects_umean_mode():
    met = {k: v for k, v in MET.items() if k != "z0"}
    fp = get_model("kljun2015")(umean=3.0, **met, **GRID)
    assert fp.attrs["wind_profile_input"] == "umean"


def test_provenance_survives_netcdf_roundtrip(tmp_path):
    pytest.importorskip("xarray")
    fp = get_model("kljun2015")(**MET, **GRID)
    path = tmp_path / "prov.nc"
    fp.to_netcdf(str(path))
    out = Footprint.from_netcdf(str(path))
    assert out.attrs["model_doi"] == fp.attrs["model_doi"]
    assert out.attrs["fluxprint_version"] == fp.attrs["fluxprint_version"]


# --------------------------------------------------------------------------- #
# CF metadata / grid mapping                                                  #
# --------------------------------------------------------------------------- #
def test_georeferenced_dataset_carries_cf_grid_mapping():
    pytest.importorskip("xarray")
    pytest.importorskip("pyproj")
    fp = Footprint(f=np.ones((5, 5)), x=np.arange(5) * 10.0 + 4.0e6,
                   y=np.arange(5) * 10.0 + 2.8e6, crs="EPSG:3035")
    ds = fp.to_xarray()
    assert ds.attrs["Conventions"] == "CF-1.8"
    assert "spatial_ref" in ds.variables
    assert ds["footprint"].attrs["grid_mapping"] == "spatial_ref"
    assert "crs_wkt" in ds["spatial_ref"].attrs
    assert ds["x"].attrs["standard_name"] == "projection_x_coordinate"


def test_local_frame_dataset_claims_no_projection_names():
    pytest.importorskip("xarray")
    ds = Footprint.from_grid(np.ones((5, 5)), dx=10.0).to_xarray()
    assert "spatial_ref" not in ds.variables
    assert "standard_name" not in ds["x"].attrs
    assert ds["x"].attrs["axis"] == "X"


def test_conventions_attr_not_duplicated_into_footprint_attrs(tmp_path):
    pytest.importorskip("xarray")
    fp = Footprint.from_grid(np.ones((5, 5)), dx=10.0)
    path = tmp_path / "conv.nc"
    fp.to_netcdf(str(path))
    out = Footprint.from_netcdf(str(path))
    assert "Conventions" not in out.attrs


def test_rioxarray_recovers_crs_from_written_netcdf(tmp_path):
    rioxarray = pytest.importorskip("rioxarray")  # noqa: F401
    xr = pytest.importorskip("xarray")
    fp = Footprint(f=np.ones((5, 5)), x=np.arange(5) * 10.0 + 4.0e6,
                   y=np.arange(5) * 10.0 + 2.8e6, crs="EPSG:3035")
    path = tmp_path / "geo.nc"
    fp.to_netcdf(str(path))
    with xr.open_dataset(str(path)) as ds:
        assert ds["footprint"].rio.crs is not None


# --------------------------------------------------------------------------- #
# Time policy                                                                 #
# --------------------------------------------------------------------------- #
def test_tz_aware_time_warns_and_converts_to_naive_utc():
    aware = datetime(2024, 4, 24, 12, 0, tzinfo=timezone.utc)
    with pytest.warns(UserWarning, match="naive UTC"):
        fp = Footprint.from_grid(np.ones((3, 3)), dx=10.0, time=aware)
    assert fp.time.tzinfo is None
    assert fp.time == datetime(2024, 4, 24, 12, 0)


def test_naive_time_passes_silently():
    fp = Footprint.from_grid(np.ones((3, 3)), dx=10.0,
                             time=datetime(2024, 4, 24, 12, 0))
    assert fp.time == datetime(2024, 4, 24, 12, 0)


# --------------------------------------------------------------------------- #
# zm = z - d (displacement height)                                            #
# --------------------------------------------------------------------------- #
def test_zm_estimated_from_measurement_and_canopy_height():
    inputs = core.process_footprint_inputs(
        measurement_height=30.0, canopy_height=15.0, z0=0.1, ustar=0.5,
        pblh=1000.0, mo_length=-100.0, v_sigma=0.5, wind_dir=180.0)
    assert inputs["zm"] == pytest.approx([30.0 - 0.67 * 15.0])
    assert "zm" in inputs.estimated


def test_explicit_displacement_takes_precedence_over_canopy_rule():
    inputs = core.process_footprint_inputs(
        measurement_height=30.0, displacement=10.0, canopy_height=15.0,
        z0=0.1, ustar=0.5, pblh=1000.0, mo_length=-100.0, v_sigma=0.5,
        wind_dir=180.0)
    assert inputs["zm"] == pytest.approx([20.0])


def test_height_columns_extracted_from_dataframe():
    import pandas as pd

    data = pd.DataFrame({
        "sensor_height": [30.0] * 2, "vegetation_height": [15.0] * 2,
        "z0": [0.1] * 2, "ustar": [0.5, 0.4], "pblh": [1000.0] * 2,
        "mo_length": [-100.0] * 2, "v_sigma": [0.5] * 2,
        "wind_dir": [180.0] * 2,
    })
    series = core.calculate_footprint(data, model="kljun2015", **GRID)
    assert series[0].n == 2
    assert "zm" in series[0].attrs["estimated_inputs"]


def test_crude_zm_constant_still_available_when_heights_absent():
    """The zm-from-heights estimator must not shadow the crude fallback."""
    with pytest.warns(UserWarning, match="crude"):
        value = micrometeorology.filler({}, "zm", fill_all=True)
    assert value == 30.0


def test_non_positive_zm_estimate_warns_and_declines():
    """displacement >= measurement height is a units/metadata mistake; the
    estimator must warn and decline instead of poisoning every record."""
    data = {"measurement_height": 5.0, "displacement": 10.0}
    with pytest.warns(UserWarning, match="non-positive zm"):
        assert micrometeorology.filler(data, "zm", fill_all=False) is None


def test_aggregate_stamps_fresh_history():
    """'history' must not depend on members computed within the same second."""
    from fluxprint.footprint import Footprint, FootprintSeries

    fps = []
    for i in range(2):
        fp = Footprint.from_grid(np.ones((3, 3)), dx=10.0, n=1, time=float(i))
        fp.attrs["history"] = f"2026-08-22T00:00:0{i}Z created ..."  # differs
        fp.attrs["model"] = "kljun2015"
        fps.append(fp)
    clim = FootprintSeries(fps).aggregate(smooth=False)
    assert "aggregated 2 footprint(s)" in clim.attrs["history"]
    assert clim.attrs["model"] == "kljun2015"
