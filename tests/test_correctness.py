"""Regression tests for the correctness fixes.

The ``exceptions`` and ``micrometeorology`` tests only need numpy. The ``core``
and ``io`` tests import modules that pull in the full geo stack (rasterio,
fiona, pyproj, xarray); they run wherever those are installed.
"""
from __future__ import annotations

import logging
import zipfile
from io import BytesIO

import numpy as np
import pytest

from fluxprint import exceptions, micrometeorology


# --------------------------------------------------------------------------- #
# exceptions.check_ffp_inputs                                                  #
# --------------------------------------------------------------------------- #
def _valid_kwargs(**overrides):
    base = dict(
        ustar=0.3, sigmav=0.5, h=1000.0, ol=-100.0, wind_dir=180.0,
        zm=10.0, z0=0.1, umean=None, rslayer=1, verbosity=0,
    )
    base.update(overrides)
    return base


def test_check_ffp_inputs_zero_ol_does_not_raise():
    """ol == 0 previously raised ZeroDivisionError; now it rejects the record."""
    assert exceptions.check_ffp_inputs(**_valid_kwargs(ol=0)) is False


def test_check_ffp_inputs_rslayer_one_continues():
    """Inside the roughness sublayer with rslayer == 1 -> alert, keep going."""
    # zm <= 12.5 * z0 puts us in the sublayer; rslayer == 1 should not reject.
    assert exceptions.check_ffp_inputs(**_valid_kwargs(zm=1.0)) is True


def test_check_ffp_inputs_rslayer_not_one_rejects():
    """Same sublayer condition but rslayer != 1 -> error, reject the record."""
    assert exceptions.check_ffp_inputs(**_valid_kwargs(zm=1.0, rslayer=0)) is False


def test_check_ffp_inputs_too_unstable_rejects():
    """zm/ol <= -15.5 is too unstable and must be rejected."""
    # zm=10, ol=-0.5 -> zm/ol = -20 <= -15.5
    assert exceptions.check_ffp_inputs(**_valid_kwargs(ol=-0.5)) is False


def test_check_ffp_inputs_rejects_nan_in_each_variable():
    """NaN passes every '<='/'>' comparison, so it needs an explicit guard."""
    for key in ("ustar", "sigmav", "h", "ol", "wind_dir", "zm", "z0"):
        assert exceptions.check_ffp_inputs(
            **_valid_kwargs(**{key: np.nan})) is False, key


def test_check_ffp_inputs_rejects_nan_umean():
    kwargs = _valid_kwargs(z0=None, umean=np.nan)
    assert exceptions.check_ffp_inputs(**kwargs) is False


def test_check_ffp_inputs_rejects_none_record():
    assert exceptions.check_ffp_inputs(**_valid_kwargs(ustar=None)) is False


def test_check_ffp_inputs_accepts_valid_record():
    assert exceptions.check_ffp_inputs(**_valid_kwargs()) is True


# --------------------------------------------------------------------------- #
# exceptions.raise_ffp_exception                                               #
# --------------------------------------------------------------------------- #
def test_fatal_code_raises_with_message_at_zero_verbosity():
    """Fatal errors used to raise Exception('') when verbosity == 0."""
    with pytest.raises(exceptions.InputValidationError, match="z0 or umean"):
        exceptions.raise_ffp_exception(15, verbosity=0)


def test_fatal_exception_is_catchable_as_fluxprint_error():
    with pytest.raises(exceptions.FluxPrintError):
        exceptions.raise_ffp_exception(1, verbosity=0)


def test_error_code_logs_warning_and_does_not_raise(caplog):
    with caplog.at_level(logging.WARNING, logger="fluxprint.exceptions"):
        exceptions.raise_ffp_exception(9, verbosity=0)
    assert any("ustar" in r.message for r in caplog.records)


def test_alert_codes_log_at_info_not_warning(caplog):
    """Routine alerts (e.g. 'Using z0') must not read like data problems."""
    with caplog.at_level(logging.INFO, logger="fluxprint.exceptions"):
        exceptions.raise_ffp_exception(13, verbosity=0)
    records = [r for r in caplog.records if "Using z0" in r.message]
    assert records and all(r.levelno == logging.INFO for r in records)


# --------------------------------------------------------------------------- #
# NaN records must not poison composited footprints                            #
# --------------------------------------------------------------------------- #
def test_calc_skips_nan_record_and_keeps_field_finite():
    """A [valid, NaN] input used to return n=2 with a zeroed/NaN field."""
    pytest.importorskip("rasterio")
    from fluxprint.model.Kljun_et_al_2015 import calc

    fp = calc(zm=[10.0, 10.0], ustar=[0.5, np.nan], pblh=[1000.0, 1000.0],
              mo_length=[-100.0, -100.0], v_sigma=[0.5, np.nan],
              wind_dir=[180.0, np.nan], z0=[0.1, 0.1],
              dx=20.0, domain=[-200, 200, -200, 200])

    assert fp.n == 1
    assert np.isfinite(fp.f).all()
    assert fp.total() > 0


def test_calculate_footprint_drops_nan_rows_from_composite():
    pytest.importorskip("rasterio")
    import pandas as pd
    from fluxprint import core

    data = pd.DataFrame({
        "zm": [10.0] * 3, "z0": [0.1] * 3, "ustar": [0.5, np.nan, 0.4],
        "pblh": [1000.0] * 3, "mo_length": [-100.0] * 3,
        "v_sigma": [0.5] * 3, "wind_dir": [180.0] * 3,
    })
    series = core.calculate_footprint(
        data, model="kljun2015", dx=20.0, domain=[-200, 200, -200, 200])

    assert series[0].n == 2                    # the NaN record is rejected
    assert np.isfinite(series[0].f).all()


# --------------------------------------------------------------------------- #
# Crude constant fills are opt-in and visible                                  #
# --------------------------------------------------------------------------- #
def test_missing_wind_dir_raises_instead_of_pointing_north():
    """wind_dir=0.0 used to be fabricated silently by default."""
    pytest.importorskip("rasterio")
    from fluxprint import core

    with pytest.raises(ValueError, match="wind_dir"):
        core.process_footprint_inputs(zm=10.0, umean=3.0, ustar=0.4,
                                      pblh=1000.0, mo_length=-100.0,
                                      v_sigma=0.5)


def test_missing_pblh_raises_without_fill_all():
    """The pblh=1000 constant used to bypass the conservative tier."""
    pytest.importorskip("rasterio")
    from fluxprint import core

    with pytest.raises(ValueError, match="pblh"):
        core.process_footprint_inputs(zm=10.0, umean=3.0, ustar=0.4,
                                      mo_length=-100.0, v_sigma=0.5,
                                      wind_dir=180.0)


def test_fill_all_warns_and_records_estimated_inputs():
    pytest.importorskip("rasterio")
    from fluxprint import core

    with pytest.warns(UserWarning, match="crude fallback"):
        inputs = core.process_footprint_inputs(
            zm=10.0, umean=3.0, ustar=0.4, mo_length=-100.0,
            fill_all=True)

    assert inputs["wind_dir"] == [0.0]
    assert inputs["pblh"] == [1000.0]
    assert "wind_dir" in inputs.estimated
    assert "pblh" in inputs.estimated


def test_processed_inputs_stay_dataframe_compatible():
    """Estimation metadata must not corrupt the returned mapping's shape."""
    pytest.importorskip("rasterio")
    import pandas as pd
    from fluxprint import core

    inputs = core.process_footprint_inputs(
        zm=[10.0] * 3, z0=[0.1] * 3, ustar=[0.5, 0.4, 0.3],
        pblh=[1000.0] * 3, mo_length=[-100.0] * 3,
        wind_dir=[180.0] * 3)  # v_sigma missing -> estimated

    frame = pd.DataFrame(inputs)      # must not raise on length mismatch
    assert len(frame) == 3
    assert "v_sigma" in inputs.estimated


def test_estimated_inputs_recorded_in_footprint_attrs():
    pytest.importorskip("rasterio")
    import pandas as pd
    from fluxprint import core

    data = pd.DataFrame({
        "zm": [10.0] * 2, "z0": [0.1] * 2, "umean": [3.0] * 2,
        "ustar": [0.5, 0.4], "pblh": [1000.0] * 2,
        "mo_length": [-100.0] * 2, "wind_dir": [180.0] * 2,
    })  # v_sigma missing -> estimated physically from ustar
    series = core.calculate_footprint(
        data, model="kljun2015", dx=20.0, domain=[-200, 200, -200, 200])

    assert "v_sigma" in series[0].attrs["estimated_inputs"]


# --------------------------------------------------------------------------- #
# z0 is accepted as the alternative to umean                                   #
# --------------------------------------------------------------------------- #
def test_z0_only_inputs_are_accepted():
    """A z0-only table used to fail with "Missing required inputs: ['umean']"."""
    pytest.importorskip("rasterio")
    from fluxprint import core

    inputs = core.process_footprint_inputs(
        zm=10.0, z0=0.1, ustar=0.4, pblh=1000.0, mo_length=-100.0,
        v_sigma=0.5, wind_dir=180.0)
    assert inputs["z0"] == [0.1]


def test_missing_both_z0_and_umean_raises():
    pytest.importorskip("rasterio")
    from fluxprint import core

    with pytest.raises(ValueError, match="z0 or umean"):
        core.process_footprint_inputs(
            zm=10.0, ustar=0.4, pblh=1000.0, mo_length=-100.0,
            v_sigma=0.5, wind_dir=180.0)


# --------------------------------------------------------------------------- #
# FLUXNET-style tables: drivers feed the estimators                            #
# --------------------------------------------------------------------------- #
def test_fluxnet_style_frame_estimates_mo_length_from_drivers():
    """USTAR/H/TA/PA (no Obukhov-length column) used to dead-end."""
    pytest.importorskip("rasterio")
    import pandas as pd
    from fluxprint import core

    n = 3
    data = pd.DataFrame({
        "USTAR": [0.4] * n, "WD": [180.0] * n, "WS": [3.0] * n,
        "H": [100.0] * n, "TA": [20.0] * n, "PA": [101.3] * n,
    })
    series = core.calculate_footprint(
        data, model="kljun2015", zm=10.0, pblh=1000.0,
        dx=20.0, domain=[-200, 200, -200, 200])

    assert series[0].n == n
    assert "mo_length" in series[0].attrs["estimated_inputs"]


def test_neutral_record_with_zero_heat_flux_is_kept():
    """H=0 gives |L| -> inf; it must clamp to the neutral limit, not be
    rejected as non-finite."""
    L = micrometeorology.compute_mo_length(0.3, 0.0, TA=20.0, PA=101.3)
    assert np.isfinite(L)
    assert abs(float(L)) >= 5000.0            # lands in the models' neutral branch


def test_zero_heat_flux_record_survives_end_to_end():
    pytest.importorskip("rasterio")
    import pandas as pd
    from fluxprint import core

    n = 2
    data = pd.DataFrame({
        "USTAR": [0.4] * n, "WD": [180.0] * n, "WS": [3.0] * n,
        "H": [150.0, 0.0], "TA": [20.0] * n, "PA": [101.3] * n,
    })
    series = core.calculate_footprint(
        data, model="kljun2015", zm=10.0, pblh=1000.0,
        dx=20.0, domain=[-200, 200, -200, 200])
    assert series[0].n == n                   # the neutral record is not dropped


def test_filler_treats_present_but_none_as_unavailable():
    """A present-but-None input used to be dereferenced and crash."""
    data = {"umean": 3.0, "ustar": 0.4, "zm": 10.0, "mo_length": None}
    assert micrometeorology.filler(data, "z0") is None


def test_explicit_none_driver_kwarg_does_not_crash():
    """H=None used to reach compute_mo_length and TypeError."""
    pytest.importorskip("rasterio")
    import pandas as pd
    from fluxprint import core

    data = pd.DataFrame({
        "zm": [10.0], "z0": [0.1], "ustar": [0.4], "v_sigma": [0.5],
        "wind_dir": [180.0],
    })
    with pytest.raises(ValueError, match="mo_length"):
        core.process_footprint_inputs(data=data, H=None, pblh=1000.0)


def test_ustar_star_alias_matches_literally_and_u_column_does_not():
    """'u*' is a literal alias, not a regex: it must match a 'u*' column and
    must NOT swallow a plain 'U' (wind component) column as friction velocity."""
    pytest.importorskip("rasterio")
    import pandas as pd
    from fluxprint import core

    base = {"zm": [10.0], "z0": [0.1], "pblh": [1000.0],
            "mo_length": [-100.0], "v_sigma": [0.5], "wind_dir": [180.0]}

    inputs = core.process_footprint_inputs(
        data=pd.DataFrame({**base, "u*": [0.4]}))
    assert inputs["ustar"] == [0.4]

    with pytest.raises(ValueError, match="ustar"):
        core.process_footprint_inputs(
            data=pd.DataFrame({**base, "U": [3.0]}),
            estimate_missing_variables=False)


def test_readme_quickstart_shape_runs():
    """The README batch example (FLUXNET columns + zm/pblh kwargs) must work."""
    pytest.importorskip("rasterio")
    import pandas as pd
    from fluxprint import core

    data = pd.DataFrame({
        "USTAR": [0.4] * 2, "WD": [180.0] * 2, "WS": [3.0] * 2,
        "H": [100.0] * 2, "TA": [20.0] * 2, "PA": [101.3] * 2,
    })
    result = core.wrapper(data=data, zm=20.0, pblh=1500.0,
                          dx=20.0, domain=[-200, 200, -200, 200])
    assert result.n == 2


def test_lowercase_h_column_is_not_mistaken_for_heat_flux():
    """'h' is FFP's name for boundary-layer height, not the heat flux H."""
    pytest.importorskip("rasterio")
    import pandas as pd
    from fluxprint import core

    data = pd.DataFrame({
        "zm": [10.0], "z0": [0.1], "ustar": [0.4], "h": [1000.0],
        "mo_length": [-100.0], "v_sigma": [0.5], "wind_dir": [180.0],
    })
    inputs = core.process_footprint_inputs(data=data, pblh=1000.0)
    assert "H" not in inputs


# --------------------------------------------------------------------------- #
# Batch smoothing matches the reference procedure (smooth once, on aggregate)  #
# --------------------------------------------------------------------------- #
def test_batch_members_are_unsmoothed_by_default():
    pytest.importorskip("rasterio")
    import pandas as pd
    from fluxprint import core
    from fluxprint.model.Kljun_et_al_2015 import calc

    row = dict(zm=10.0, z0=0.1, umean=3.0, ustar=0.5, pblh=1000.0,
               mo_length=-100.0, v_sigma=0.5, wind_dir=180.0)
    data = pd.DataFrame({k: [v] for k, v in row.items()})
    grid = dict(dx=20.0, domain=[-200, 200, -200, 200])

    series = core.calculate_footprint(data, model="kljun2015", **grid)
    unsmoothed = calc(**row, smooth_data=0, **grid)
    smoothed = calc(**row, smooth_data=1, **grid)

    assert np.allclose(series[0].f, unsmoothed.f)
    assert not np.allclose(series[0].f, smoothed.f)


# --------------------------------------------------------------------------- #
# NetCDF-safe attrs: the by-timestamp workflow must serialize                  #
# --------------------------------------------------------------------------- #
def test_timestamp_group_labels_serialize_to_netcdf(tmp_path):
    pytest.importorskip("rasterio")
    pytest.importorskip("xarray")
    import pandas as pd
    from fluxprint import core

    data = pd.DataFrame({
        "t": pd.to_datetime(["2024-04-24 00:00", "2024-04-24 00:30"]),
        "zm": [10.0] * 2, "z0": [0.1] * 2, "ustar": [0.5, 0.4],
        "pblh": [1000.0] * 2, "mo_length": [-100.0] * 2,
        "v_sigma": [0.5] * 2, "wind_dir": [180.0] * 2,
    })
    series = core.calculate_footprint(
        data, by="t", model="kljun2015", dx=20.0,
        domain=[-200, 200, -200, 200])

    assert all(isinstance(fp.attrs["group"], str) for fp in series)
    series.to_netcdf(tmp_path / "series.nc")       # used to raise TypeError
    series[0].to_netcdf(tmp_path / "single.nc")


# --------------------------------------------------------------------------- #
# Captured-fraction diagnostic                                                 #
# --------------------------------------------------------------------------- #
def test_captured_fraction_recorded_and_warns_on_truncation(caplog):
    pytest.importorskip("rasterio")
    from fluxprint.model.Kljun_et_al_2015 import calc

    with caplog.at_level(logging.WARNING,
                         logger="fluxprint.model.kljun_et_al_2015"):
        fp = calc(zm=20.0, z0=0.1, ustar=0.5, pblh=1000.0, mo_length=-100.0,
                  v_sigma=0.5, wind_dir=180.0, dx=10.0,
                  domain=[-500, 500, -500, 500])

    assert fp.attrs["captured_fraction"] == pytest.approx(fp.total())
    assert fp.attrs["captured_fraction"] < 0.8
    assert any("captures only" in r.message for r in caplog.records)


# --------------------------------------------------------------------------- #
# micrometeorology.caller                                                      #
# --------------------------------------------------------------------------- #
def test_caller_pblh_is_lazy_and_does_not_require_z0_inputs():
    """Requesting pblh used to KeyError because z0 was computed eagerly."""
    assert micrometeorology.caller({}, "pblh") == 1000.0


def test_caller_v_sigma_only_needs_ustar():
    out = micrometeorology.caller({"ustar": [0.2, 0.4]}, "v_sigma")
    assert np.allclose(out, [0.4, 0.8])  # sigma_v = 2.0 * ustar


def test_caller_unknown_variable_returns_none():
    assert micrometeorology.caller({}, "no_such_variable") is None


def test_crude_constant_emits_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="fluxprint.micrometeorology"):
        value = micrometeorology.filler({}, "zm", fill_all=True)
    assert value == 30.0
    assert any("crude" in r.message for r in caplog.records)


def test_fill_all_toggles_crude_estimators():
    # zm is a crude constant: available only when fill_all=True.
    assert micrometeorology.filler({}, "zm", fill_all=True) == 30.0
    assert micrometeorology.filler({}, "zm", fill_all=False) is None
    # v_sigma is an essential estimate: available in both tiers.
    assert micrometeorology.filler({"ustar": 0.3}, "v_sigma", fill_all=False) is not None


def test_filler_returns_none_when_inputs_unavailable():
    # mo_length needs ustar + H + TA + PA; with nothing it can't compute.
    assert micrometeorology.filler({}, "mo_length", fill_all=True) is None


# --------------------------------------------------------------------------- #
# core.calculate_footprint  (full geo stack)                                   #
# --------------------------------------------------------------------------- #
def test_calculate_footprint_skips_failed_group():
    """A failed group is skipped, never backfilled with another group's footprint."""
    pytest.importorskip("rasterio")
    import pandas as pd
    from fluxprint import core
    from fluxprint.footprint import Footprint, FootprintSeries

    calls = {"n": 0}

    def fake(*, dx, time=None, **kw):
        calls["n"] += 1
        if calls["n"] == 2:          # groups iterate sorted: A ok, B fails
            raise RuntimeError("boom")
        return Footprint.from_grid(np.zeros((3, 3)), dx=dx, time=time, n=1)

    n = 2
    data = pd.DataFrame({
        "grp": ["A", "B"],
        "zm": [10.0] * n, "umean": [3.0] * n, "ustar": [0.3] * n,
        "pblh": [1000.0] * n, "mo_length": [-100.0] * n,
        "v_sigma": [0.5] * n, "wind_dir": [180.0] * n,
    })

    series = core.calculate_footprint(data, by="grp", model=fake)

    assert isinstance(series, FootprintSeries)
    assert series.nt == 1                       # B dropped, not duplicated
    assert series[0].attrs["group"] == "A"


def test_calculate_footprint_raises_when_all_groups_fail():
    pytest.importorskip("rasterio")
    import pandas as pd
    from fluxprint import core

    def failing(**kw):
        raise RuntimeError("boom")

    data = pd.DataFrame({
        "zm": [10.0], "umean": [3.0], "ustar": [0.3], "pblh": [1000.0],
        "mo_length": [-100.0], "v_sigma": [0.5], "wind_dir": [180.0],
    })

    with pytest.raises(ValueError, match="No footprints"):
        core.calculate_footprint(data, model=failing)


# --------------------------------------------------------------------------- #
# io.read_from_url  (full geo stack)                                           #
# --------------------------------------------------------------------------- #
class _FakeResponse:
    def __init__(self, content: bytes, status_code: int = 200):
        self.content = content
        self.status_code = status_code


def test_read_from_url_raises_on_http_error(monkeypatch):
    pytest.importorskip("rasterio")
    from fluxprint import io

    monkeypatch.setattr(io.requests, "get",
                        lambda *a, **k: _FakeResponse(b"", status_code=404))
    with pytest.raises(OSError, match="404"):
        io.read_from_url("http://example.invalid/data.zip")


def test_read_from_url_raises_on_unparseable_payload(monkeypatch):
    pytest.importorskip("rasterio")
    from fluxprint import io

    monkeypatch.setattr(io.requests, "get",
                        lambda *a, **k: _FakeResponse(b"not a zip or netcdf"))
    with pytest.raises(ValueError, match="zip"):
        io.read_from_url("http://example.invalid/data.bin")


def test_read_from_url_reads_zipped_csv(monkeypatch):
    pytest.importorskip("rasterio")
    from fluxprint import io

    buf = BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("data.csv", "a,b\n1,2\n3,4\n")
    monkeypatch.setattr(io.requests, "get",
                        lambda *a, **k: _FakeResponse(buf.getvalue()))

    df = io.read_from_url("http://example.invalid/data.zip")
    assert list(df.columns) == ["a", "b"]
    assert df["b"].tolist() == [2, 4]
