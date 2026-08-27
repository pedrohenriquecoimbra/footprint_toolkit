"""Tests for fluxprint.model.engine: the generic model pipeline.

A toy Gaussian kernel registered via ``@footprint_model`` must get the whole
driver for free — input normalization, grid, climatology loop, smoothing,
provenance, ``captured_fraction`` — and flow through ``calculate_footprint``
like any built-in model.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("rasterio")  # the package import pulls the geo stack

from fluxprint.exceptions import InputValidationError  # noqa: E402
from fluxprint.footprint import Footprint, smooth_field  # noqa: E402
from fluxprint.grid import resolve_grid  # noqa: E402
from fluxprint.model import (MODELS, FootprintModel, available_models,  # noqa: E402
                             footprint_model, get_model)
from fluxprint.model.engine import listify, normalize_inputs  # noqa: E402

SIGMA = 60.0

MET = dict(zm=20.0, ustar=0.5, pblh=1000.0, mo_length=-100.0, v_sigma=0.5,
           wind_dir=0.0, z0=0.1)
GRID = dict(domain=[-300.0, 300.0, -300.0, 300.0], dx=10.0)


def _gauss_field(ctx):
    return (np.exp(-(ctx.x_2d**2 + ctx.y_2d**2) / (2 * SIGMA**2))
            / (2 * np.pi * SIGMA**2))


@pytest.fixture
def toy_model():
    @footprint_model("toy_gauss", description="toy Gaussian",
                     meta={"model_doi": "10.0000/toy"},
                     options=("width",), defaults={"width": 1.0})
    def kernel(ctx, rec, opts):
        return _gauss_field(ctx), 0, 1

    try:
        yield get_model("toy_gauss")
    finally:
        MODELS.pop("toy_gauss", None)


@pytest.fixture
def flaky_model():
    # Stable records (mo_length > 0) turn invalid mid-computation (flag 3).
    @footprint_model("toy_flaky", validate=lambda rec, opts, verbosity: True)
    def kernel(ctx, rec, opts):
        if rec.mo_length > 0:
            return np.zeros(ctx.x_2d.shape), 3, 0
        return _gauss_field(ctx), 0, 1

    try:
        yield get_model("toy_flaky")
    finally:
        MODELS.pop("toy_flaky", None)


# --------------------------------------------------------------------------- #
# normalize_inputs / listify units                                            #
# --------------------------------------------------------------------------- #
def test_listify_converts_sequences_and_passes_scalars():
    assert listify(np.array([1.0, 2.0])) == [1.0, 2.0]
    assert listify((1.0, 2.0)) == [1.0, 2.0]
    assert listify(3.0) == 3.0
    assert listify(None) is None


def test_normalize_inputs_broadcasts_zm_and_prefers_z0():
    out = normalize_inputs(zm=[20.0], ustar=[0.5, 0.4], pblh=[1000.0] * 2,
                           mo_length=[-100.0] * 2, v_sigma=[0.5] * 2,
                           wind_dir=[0.0, 90.0], z0=[0.1], umean=[3.0, 3.0],
                           verbosity=0)
    assert out.ts_len == 2
    assert out.zm == [20.0, 20.0]
    assert out.z0 == [0.1, 0.1]
    assert out.umean == [None, None]  # z0 wins when both are given


def test_normalize_inputs_rejects_unequal_lengths():
    with pytest.raises(InputValidationError):
        normalize_inputs(zm=[20.0], ustar=[0.5, 0.4], pblh=[1000.0],
                         mo_length=[-100.0], v_sigma=[0.5], wind_dir=[0.0],
                         z0=[0.1], verbosity=0)


# --------------------------------------------------------------------------- #
# @footprint_model: registration and the free driver                          #
# --------------------------------------------------------------------------- #
def test_decorator_registers_a_protocol_conformant_model(toy_model):
    assert "toy_gauss" in available_models()
    assert isinstance(toy_model, FootprintModel)
    assert toy_model.resolve_grid is resolve_grid
    assert callable(toy_model.kernel)


def test_toy_model_gets_grid_provenance_and_captured_fraction(toy_model):
    fp = toy_model(**MET, **GRID, smooth_data=0, verbosity=0)
    assert isinstance(fp, Footprint)
    assert fp.f.shape == (61, 61)
    assert fp.n == 1
    assert fp.attrs["model"] == "toy_gauss"
    assert fp.attrs["model_doi"] == "10.0000/toy"
    assert "fluxprint_version" in fp.attrs
    assert "toy_gauss" in fp.attrs["history"]
    assert fp.attrs["captured_fraction"] == pytest.approx(fp.total())
    assert fp.attrs["smooth_data"] == 0


def test_toy_model_smoothing_uses_the_generic_kernel(toy_model):
    raw = toy_model(**MET, **GRID, smooth_data=0, verbosity=0)
    smoothed = toy_model(**MET, **GRID, smooth_data=1, verbosity=0)
    assert np.array_equal(smoothed.f, smooth_field(raw.f))
    assert smoothed.attrs["smooth_data"] == 1


def test_toy_model_composites_records(toy_model):
    met = {k: [v] * 3 for k, v in MET.items()}
    fp = toy_model(**met, **GRID, smooth_data=0, verbosity=0)
    assert fp.n == 3
    # mean of three identical fields is the field itself
    single = toy_model(**MET, **GRID, smooth_data=0, verbosity=0)
    assert np.allclose(fp.f, single.f)


def test_default_validation_skips_implausible_records(toy_model):
    met = {k: [v] * 3 for k, v in MET.items()}
    met["ustar"] = [0.5, 0.05, 0.5]  # ustar <= 0.1 fails the FFP checks
    fp = toy_model(**met, **GRID, smooth_data=0, verbosity=0)
    assert fp.n == 2


def test_model_options_reach_kernel_and_attrs():
    seen = {}

    @footprint_model("toy_opts", validate=lambda rec, opts, verbosity: True,
                     options=("width",), defaults={"width": 1.0})
    def kernel(ctx, rec, opts):
        seen.update(opts)
        return _gauss_field(ctx) * opts["width"], 0, 1

    try:
        model = get_model("toy_opts")
        fp = model(**MET, **GRID, smooth_data=0, verbosity=0, width=2.0)
        assert seen["width"] == 2.0
        assert fp.attrs["width"] == 2.0
        assert fp.attrs["captured_fraction"] == pytest.approx(2.0, rel=0.05)
    finally:
        MODELS.pop("toy_opts", None)


# --------------------------------------------------------------------------- #
# flag_err bookkeeping (reference-exact precedence)                           #
# --------------------------------------------------------------------------- #
def test_flag3_latches_when_other_records_are_valid(flaky_model):
    met = {k: [v] * 2 for k, v in MET.items()}
    met["mo_length"] = [-100.0, 200.0]  # second record goes flag-3 invalid
    fp = flaky_model(**met, **GRID, smooth_data=0, verbosity=0)
    assert fp.n == 1
    assert fp.attrs["flag_err"] == 3


def test_all_invalid_overwrites_flag_to_1(flaky_model):
    fp = flaky_model(**{**MET, "mo_length": 200.0}, **GRID, smooth_data=0,
                     verbosity=0)
    assert fp.n == 0
    assert fp.attrs["flag_err"] == 1  # n==0 overwrites the latched 3
    assert "captured_fraction" not in fp.attrs


# --------------------------------------------------------------------------- #
# End to end through the batch layer                                          #
# --------------------------------------------------------------------------- #
def test_toy_model_flows_through_calculate_footprint(toy_model):
    from fluxprint import calculate_footprint

    data = {k: [v] * 4 for k, v in MET.items()}
    series = calculate_footprint(data=data, model="toy_gauss", **GRID)
    assert len(series) == 1
    assert series[0].n == 4
    assert series[0].attrs["model"] == "toy_gauss"


# --------------------------------------------------------------------------- #
# smooth / smooth_data: one knob, two spellings                               #
# --------------------------------------------------------------------------- #
def test_smooth_spellings_are_equivalent_on_kljun():
    kljun = get_model("kljun2015")
    via_generic = kljun(**MET, **GRID, smooth=0, verbosity=0)
    via_ffp = kljun(**MET, **GRID, smooth_data=0, verbosity=0)
    assert np.array_equal(via_generic.f, via_ffp.f)
    assert via_generic.attrs["smooth_data"] == 0


def test_smooth_wins_over_smooth_data(toy_model):
    both = toy_model(**MET, **GRID, smooth=0, smooth_data=1, verbosity=0)
    off = toy_model(**MET, **GRID, smooth_data=0, verbosity=0)
    assert np.array_equal(both.f, off.f)


# --------------------------------------------------------------------------- #
# calc_ffp_climatology: deprecated crop/rs kwargs                             #
# --------------------------------------------------------------------------- #
def test_shim_warns_on_deprecated_crop_and_rs():
    from fluxprint.model.Kljun_et_al_2015 import calc_ffp_climatology

    met = {k: [v] for k, v in MET.items()}
    with pytest.warns(DeprecationWarning, match="crop"):
        out = calc_ffp_climatology(**met, **GRID, crop=1, verbosity=0)
    assert out.n == 1  # crop is ignored, the field is still computed
    with pytest.warns(DeprecationWarning, match="rs"):
        calc_ffp_climatology(**met, **GRID, rs=[0.8], verbosity=0)
