"""Model-agnostic driver machinery for footprint models.

Everything here used to live inside the Kljun port and was copy-pasted (with
divergence) by every new model draft. The pieces are pipeline functions a
model adapter composes — a model itself is only physics: a per-record kernel
evaluated on a :class:`~fluxprint.grid.GridContext`.

The registered Kljun model is pinned **bitwise** to the vendored reference
implementation (tests/test_reference_regression.py), so every function moved
here is verbatim code motion: same float operations, same order, same dtypes,
quirks included. Do not "improve" anything on the numeric path.

This module reserves the seam the planned xarray/dask compute path needs: a
pure kernel plus :attr:`fluxprint.grid.GridSpec.shape` gives the output shape
without compute, and a kernel neither raises nor logs per record.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, NamedTuple

import numpy as np

from ..exceptions import check_ffp_inputs, raise_ffp_exception
from ..footprint import Footprint, smooth_field

logger = logging.getLogger('fluxprint.model.engine')

__all__ = ["NormalizedInputs", "MetRecord", "ClimResult", "listify",
           "normalize_inputs", "ffp_validate", "run_climatology",
           "build_footprint"]


class NormalizedInputs(NamedTuple):
    """Equal-length per-record input lists plus their common length.

    Values are passed through untouched — no float()/numpy coercion — so a
    kernel sees exactly the objects the caller provided (bit patterns
    matter on the pinned reference path).
    """

    zm: list
    ustar: list
    pblh: list
    mo_length: list
    v_sigma: list
    wind_dir: list
    z0: list
    umean: list
    ts_len: int


def listify(value: Any) -> Any:
    """Arrays / tuples / Series -> list; scalars and None pass through.

    :func:`normalize_inputs` wraps non-lists as a single element (the FFP
    convention), so sequence containers must become real lists first.
    """
    if value is None or isinstance(value, (int, float, list)):
        return value
    return list(value)


def normalize_inputs(*, zm, ustar, pblh, mo_length, v_sigma, wind_dir,
                     z0=None, umean=None, verbosity=0) -> NormalizedInputs:
    """Validate and broadcast the met inputs into equal-length lists.

    Verbatim move of the FFP reference's input block: presence check
    (exception code 1), scalar wrapping, equal-length check (11), the
    length-1 ``zm`` broadcast (12/17), and the ``z0``-vs-``umean``
    precedence resolution (13/14/15; ``z0`` wins when both are given).
    Callers with array/Series inputs should map :func:`listify` over them
    first (the FFP convention wraps any non-list as one element).
    """
    # Check existence of required input pars
    if None in [zm, pblh, mo_length, v_sigma, ustar] or (z0 is None and umean is None):
        raise_ffp_exception(1, verbosity)

    # Convert all input items to lists
    if not isinstance(zm, list): zm = [zm]
    if not isinstance(pblh, list): pblh = [pblh]
    if not isinstance(mo_length, list): mo_length = [mo_length]
    if not isinstance(v_sigma, list): v_sigma = [v_sigma]
    if not isinstance(ustar, list): ustar = [ustar]
    if not isinstance(wind_dir, list): wind_dir = [wind_dir]
    if not isinstance(z0, list): z0 = [z0]
    if not isinstance(umean, list): umean = [umean]

    # Check that all lists have same length, if not raise an error and exit
    ts_len = len(ustar)
    if any(len(lst) != ts_len for lst in [v_sigma, wind_dir, pblh, mo_length]):
        # at least one list has a different length, exit with error message
        raise_ffp_exception(11, verbosity)

    # Special treatment for zm, which is allowed to have length 1 for any
    # length >= 1 of all other parameters
    if all(val is None for val in zm): raise_ffp_exception(12, verbosity)
    if len(zm) == 1:
        raise_ffp_exception(17, verbosity)
        zm = [zm[0] for i in range(ts_len)]

    # Resolve ambiguity if both z0 and umean are passed (defaults to using z0)
    # If at least one value of z0 is passed, use z0 (by setting umean to None)
    if not all(val is None for val in z0):
        raise_ffp_exception(13, verbosity)
        umean = [None for i in range(ts_len)]
        # If only one value of z0 was passed, use that value for all footprints
        if len(z0) == 1: z0 = [z0[0] for i in range(ts_len)]
    elif len(umean) == ts_len and not all(val is None for val in umean):
        raise_ffp_exception(14, verbosity)
        z0 = [None for i in range(ts_len)]
    else:
        raise_ffp_exception(15, verbosity)

    return NormalizedInputs(zm=zm, ustar=ustar, pblh=pblh,
                            mo_length=mo_length, v_sigma=v_sigma,
                            wind_dir=wind_dir, z0=z0, umean=umean,
                            ts_len=ts_len)


class MetRecord(NamedTuple):
    """One record's met inputs, exactly as provided (no coercion)."""

    ustar: Any
    v_sigma: Any
    pblh: Any
    mo_length: Any
    wind_dir: Any
    zm: Any
    z0: Any
    umean: Any


class ClimResult(NamedTuple):
    """Raw climatology output of :func:`run_climatology`."""

    fclim_2d: np.ndarray
    x: np.ndarray
    y: np.ndarray
    n: int
    flag_err: int


def ffp_validate(rec: MetRecord, opts: dict, verbosity: int) -> bool:
    """Default per-record validation: the FFP physical-plausibility checks."""
    return check_ffp_inputs(rec.ustar, rec.v_sigma, rec.pblh, rec.mo_length,
                            rec.wind_dir, rec.zm, rec.z0, rec.umean,
                            opts.get("rslayer", 0), verbosity)


def run_climatology(kernel: Callable, *, ctx, inputs: NormalizedInputs,
                    opts: dict | None = None, validate: Callable = ffp_validate,
                    smooth_data=1, pulse=None, verbosity=0) -> ClimResult:
    """The model-agnostic climatology loop, hoisted verbatim from the port.

    Per record: validate (invalid records are skipped with exception code
    16), evaluate ``kernel(ctx, rec, opts) -> (f_2d, flag, valid)``,
    accumulate. Then normalize the accumulated field by the number of valid
    records and, when ``smooth_data`` is truthy, apply the standard FFP
    smoothing (:func:`~fluxprint.footprint.smooth_field`).

    ``flag_err`` bookkeeping reproduces the reference exactly: a nonzero
    kernel flag (3: a record turned invalid mid-computation) latches, and
    zero valid records *overwrites* it with 1.

    Args:
        kernel: Per-record model physics. Must not raise for per-record
            invalidity (return ``valid=0`` instead), must not log, and must
            treat ``ctx`` arrays as read-only.
        ctx: :class:`~fluxprint.grid.GridContext` for the output grid.
        inputs: :func:`normalize_inputs` output.
        opts: Model-specific options passed through to ``kernel``/``validate``
            (e.g. ``{"rslayer": 1}``).
        validate: ``(rec, opts, verbosity) -> bool`` per-record gate.
        smooth_data: Truthy to smooth the final climatology (FFP default).
        pulse: Progress-log cadence; defaults to ~5% of the series.
        verbosity: 2 logs progress, 1 only fatal problems, 0 silent.
    """
    opts = {} if opts is None else opts
    flag_err = 0
    fclim_2d = np.zeros(ctx.x_2d.shape)
    ts_len = inputs.ts_len

    # Define pulse if not passed
    if pulse == None:
        if ts_len <= 20:
            pulse = 1
        else:
            pulse = int(ts_len / 20)

    # Initialize logic array valids to those 'timestamps' for which all inputs are
    # at least present (but not necessarily phisically plausible)
    valids = [True if not any([val is None for val in vals]) else False
              for vals in zip(inputs.ustar, inputs.v_sigma, inputs.pblh,
                              inputs.mo_length, inputs.wind_dir, inputs.zm)]

    records = map(MetRecord._make,
                  zip(inputs.ustar, inputs.v_sigma, inputs.pblh,
                      inputs.mo_length, inputs.wind_dir, inputs.zm,
                      inputs.z0, inputs.umean))
    for ix, rec in enumerate(records):
        # Counter
        if verbosity > 1 and ix % pulse == 0:
            logger.info('Calculating footprint %d of %d', ix + 1, ts_len)

        valids[ix] = validate(rec, opts, verbosity)

        # If inputs are not valid, skip current footprint
        if not valids[ix]:
            raise_ffp_exception(16, verbosity)
        else:
            f_2d, flag, valid = kernel(ctx, rec, opts)
            if flag:
                flag_err = flag
            if not valid:
                valids[ix] = 0
            # Add to footprint climatology raster
            fclim_2d = fclim_2d + f_2d

    #===========================================================================
    # Continue if at least one valid footprint was calculated
    n = sum(valids)
    if n == 0:
        logger.error("No footprint calculated")
        flag_err = 1
    else:
        logger.info(f"{n} footprint calculated")
        # Normalize and smooth footprint climatology
        fclim_2d = fclim_2d / n

        # Truthiness, not `is not None`: smooth_data=0 must actually disable
        # smoothing as documented (it used to smooth anyway).
        if smooth_data:
            fclim_2d = smooth_field(fclim_2d)

    return ClimResult(fclim_2d=fclim_2d, x=ctx.x, y=ctx.y, n=n,
                      flag_err=flag_err)


def build_footprint(result: ClimResult, *, name: str,
                    meta: dict | None = None, wind_profile_input: str,
                    settings: dict | None = None, tower=None, tower_crs=None,
                    time=None, log: logging.Logger | None = None) -> Footprint:
    """Wrap a raw climatology into a provenance-stamped :class:`Footprint`.

    The adapter tail every model used to copy: provenance attrs (model name,
    error flag, the model's citation ``meta``, fluxprint version, settings,
    a ``history`` line) plus the ``captured_fraction`` diagnostic with its
    under-capture warning (emitted on ``log`` so it stays attributed to the
    producing model's logger).

    Args:
        result: :func:`run_climatology` output.
        name: Registered model name (stamped into ``attrs["model"]``).
        meta: The model's provenance dict (citation/DOI/reference version).
        wind_profile_input: ``"z0"`` or ``"umean"`` — which profile input
            drove the calculation.
        settings: Model settings worth recording (e.g. ``rslayer``,
            ``smooth_data``), inserted before the ``history`` line.
        tower, tower_crs, time: Metadata attached to the footprint.
        log: Logger for the truncation warning; defaults to the engine's.
    """
    from datetime import datetime as _dt, timezone as _tz

    from ..version import __version__ as _fluxprint_version

    log = logger if log is None else log
    attrs = {
        "model": name,
        "flag_err": int(result.flag_err),
        **(meta or {}),
        "fluxprint_version": _fluxprint_version,
        "wind_profile_input": wind_profile_input,
        **(settings or {}),
        "history": (f"{_dt.now(_tz.utc).strftime('%Y-%m-%dT%H:%M:%SZ')} "
                    f"created by fluxprint {_fluxprint_version}, "
                    f"model {name}"),
    }
    fp = Footprint(
        f=np.asarray(result.fclim_2d), x=np.asarray(result.x),
        y=np.asarray(result.y), time=time,
        tower=tower, tower_crs=tower_crs, n=int(result.n), attrs=attrs)
    if fp.n:
        # The grid integral of the full footprint is 1 by construction, so the
        # captured fraction diagnoses how much flux the domain truncates. It is
        # computed on the returned field, so with smoothing on it also
        # includes the ~1% border mass the smoothing kernel loses.
        captured = fp.total()
        fp.attrs["captured_fraction"] = float(captured)
        if captured < 0.8:
            log.warning(
                "Footprint domain captures only %.0f%% of the flux; source-"
                "area fractions above that are unreachable. Enlarge `domain` "
                "(or reduce `dx`) to capture more.", captured * 100)
    return fp
