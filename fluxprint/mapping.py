"""xarray/dask-native footprint computation.

:func:`map_footprints` maps a footprint model's per-record kernel elementwise
over array inputs of any dimensionality and returns the footprint field as a
``DataArray`` with dims ``(*input_dims, y, x)``. With dask-backed inputs the
result is lazy and each chunk is computed independently — the shape is known
in advance from the model's :class:`~fluxprint.grid.GridSpec`, the kernel is
a pure function that neither raises nor logs per record (rejections are
summarised once per block), and a bad record costs a NaN plane, not the job.

This is the compute-kernel counterpart of :func:`~fluxprint.core.
calculate_footprint`: no pandas grouping, no input estimation, no
climatology — one footprint per record, exactly as a downstream pipeline
(e.g. fluxcom's provider) consumes it. Compose reductions downstream
(``result.mean("time")``, :meth:`Footprint.weighted_mean` on slices, ...).

Requires ``xarray`` (imported lazily); dask is optional and only needed for
lazy evaluation.
"""
from __future__ import annotations

import inspect
import logging
import re

import numpy as np

from .core import ALIASES, _resolve_model
from .exceptions import InputValidationError
from .footprint import smooth_field
from .grid import GridContext, resolve_grid as _default_resolve_grid
from .model import engine

logger = logging.getLogger("fluxprint.mapping")

__all__ = ["map_footprints"]

#: Required met variables of the canonical model signature, in kernel order.
_MET_VARS = ("zm", "ustar", "pblh", "mo_length", "v_sigma", "wind_dir")


def _find_variable(data, name):
    """Look up ``name`` in a Dataset by canonical name, then aliases.

    Exact canonical match first, then each spelling from
    ``ALIASES["model_inputs"]`` matched case-insensitively — the same
    resolution order :func:`~fluxprint.core.process_footprint_inputs` uses.
    """
    if data is None:
        return None
    names = list(data.data_vars)
    if name in names:
        return data[name]
    for cand in (name, *ALIASES["model_inputs"].get(name, ())):
        pattern = re.compile(f"^{re.escape(cand)}$", re.IGNORECASE)
        for col in names:
            if pattern.match(col):
                return data[col]
    return None


def map_footprints(data=None, model="kljun2015", *, domain=None, dx=None,
                   dy=None, nx=None, ny=None, smooth=0, on_error="nan",
                   **inputs):
    """One footprint field per record, mapped over arrays of any shape.

    Args:
        data: ``xarray.Dataset`` holding the met variables (canonical names
            ``zm``/``ustar``/``pblh``/``mo_length``/``v_sigma``/``wind_dir``
            plus ``z0`` or ``umean``; the spellings in
            ``fluxprint.ALIASES["model_inputs"]`` are recognized). Variables
            may span any dims — ``(time,)``, ``(site, time, hour)``, ... —
            and broadcast against each other; dask-backed variables keep the
            result lazy.
        model: Registered model name or callable. Must expose a per-record
            kernel (``kljun2015`` and every ``@footprint_model``-built model
            do).
        domain, dx, dy, nx, ny: Output grid, resolved by the model's grid
            rules (:func:`~fluxprint.grid.resolve_grid` defaults).
        smooth: Truthy to smooth each record's field (off by default — this
            is a compute kernel; smooth climatologies, not members).
        on_error: ``"nan"`` (default) turns an invalid record — or one the
            kernel fails on — into an all-NaN plane, with one summary log
            line per block; ``"raise"`` aborts on the first bad record.
        **inputs: Met variables passed directly (scalars or DataArrays;
            override ``data``), plus model options (e.g. ``rslayer=1``).

    Returns:
        ``xarray.DataArray`` named ``footprint``, dims ``(*input_dims, y,
        x)``, coords from the resolved grid, units m-2.
    """
    import xarray as xr

    if on_error not in ("nan", "raise"):
        raise ValueError(
            f"on_error must be 'nan' or 'raise'; got {on_error!r}.")
    model_fn = _resolve_model(model)
    kernel = getattr(model_fn, "kernel", None)
    if kernel is None:
        raise TypeError(
            f"model {getattr(model_fn, '__name__', model)!r} does not expose "
            "a per-record kernel; the mapped path needs a kernel-based model "
            "(register one with @footprint_model, or use "
            "calculate_footprint).")
    validate = getattr(model_fn, "validate", engine.ffp_validate)
    quiet_kw = ({"quiet": True}
                if "quiet" in inspect.signature(validate).parameters else {})

    opts = dict(getattr(model_fn, "option_defaults", {}))
    for key in getattr(model_fn, "model_options", ()):
        if inputs.get(key) is not None:
            opts[key] = inputs.pop(key)

    if data is not None and not hasattr(data, "data_vars"):
        raise TypeError(
            f"Unsupported `data` container: {type(data).__name__}. "
            "map_footprints takes an xarray.Dataset (or met variables as "
            "keyword arguments); for tabular input use calculate_footprint.")

    met = {}
    for name in _MET_VARS + ("z0", "umean"):
        value = inputs.pop(name, None)
        met[name] = value if value is not None else _find_variable(data, name)
    if inputs:
        raise TypeError(f"Unexpected keyword argument(s): {sorted(inputs)}.")
    missing = [n for n in _MET_VARS if met[n] is None]
    if missing:
        raise ValueError(
            f"Missing required met variable(s): {missing}. Provide them in "
            "`data` or as keyword arguments (recognized spellings: "
            "fluxprint.ALIASES['model_inputs']).")
    if met["z0"] is None and met["umean"] is None:
        raise ValueError("Either z0 or umean is required.")
    profile = "z0" if met["z0"] is not None else "umean"
    aux = met[profile]

    resolver = getattr(model_fn, "resolve_grid", _default_resolve_grid)
    spec = resolver(domain=domain, dx=dx, dy=dy, nx=nx, ny=ny)
    ctx = GridContext(spec)

    def _block(zm_a, ustar_a, pblh_a, ol_a, sv_a, wd_a, aux_a):
        arrs = np.broadcast_arrays(zm_a, ustar_a, pblh_a, ol_a, sv_a, wd_a,
                                   aux_a)
        shape = arrs[0].shape
        out = np.empty(shape + ctx.spec.shape, dtype=float)
        rejected = 0
        for idx in np.ndindex(shape):
            rec = engine.MetRecord(
                ustar=arrs[1][idx], v_sigma=arrs[4][idx], pblh=arrs[2][idx],
                mo_length=arrs[3][idx], wind_dir=arrs[5][idx],
                zm=arrs[0][idx],
                z0=arrs[6][idx] if profile == "z0" else None,
                umean=arrs[6][idx] if profile == "umean" else None)
            field = None
            try:
                if validate(rec, opts, 0, **quiet_kw):
                    f_2d, _flag, valid = kernel(ctx, rec, opts)
                    if valid:
                        field = f_2d
            except Exception:
                if on_error == "raise":
                    raise
            if field is None:
                if on_error == "raise":
                    raise InputValidationError(
                        f"record {idx}: invalid input for the footprint "
                        "model (on_error='raise').")
                out[idx] = np.nan
                rejected += 1
            else:
                out[idx] = smooth_field(field) if smooth else field
        if rejected:
            logger.warning(
                "map_footprints: %d of %d record(s) invalid -> NaN plane.",
                rejected, max(int(np.prod(shape)), 1))
        return out

    result = xr.apply_ufunc(
        _block,
        met["zm"], met["ustar"], met["pblh"], met["mo_length"],
        met["v_sigma"], met["wind_dir"], aux,
        output_core_dims=[["y", "x"]],
        dask="parallelized",
        output_dtypes=[float],
        dask_gufunc_kwargs={"output_sizes": {"y": spec.shape[0],
                                             "x": spec.shape[1]}},
    )
    x, y = spec.axes()
    result = result.assign_coords(x=("x", x), y=("y", y))
    result.name = "footprint"

    from .version import __version__
    result.attrs.update({
        "units": "m-2",
        "long_name": "flux footprint",
        "model": model if isinstance(model, str)
        else getattr(model_fn, "__name__", type(model_fn).__name__),
        "fluxprint_version": __version__,
        "wind_profile_input": profile,
        "smooth": int(bool(smooth)),
        "on_error": on_error,
        "domain": list(spec.domain),
        "dx": spec.dx, "dy": spec.dy,
    })
    return result
