# built-in modules
import warnings
import re
import numbers
import logging
from datetime import datetime

# 3rd party modules
import numpy as np
import pandas as pd
from scipy import signal as sg

# local modules (deliberately light: the geo/plotting stack loads lazily via
# `utils`/`io` function-level imports, so `import fluxprint` stays cheap)
from .footprint import Footprint, FootprintSeries, smooth_field
from .model import get_model
from . import io
from . import exceptions
from . import micrometeorology

logger = logging.getLogger('fluxprint.core')


#: Alternative column names recognized by :func:`process_footprint_inputs`,
#: keyed by canonical name. ``"model_inputs"`` are the met variables forwarded
#: to the models; ``"drivers"`` are consumed only by the estimators (e.g. the
#: Obukhov length from USTAR/H/TA/PA) and each entry lists every accepted
#: spelling including the canonical one. Matching is case-insensitive except
#: for ``'H'`` (sensible heat flux): FFP names the boundary-layer height with
#: a lowercase ``'h'``, so ``'H'`` must match exactly.
ALIASES = {
    "model_inputs": {
        "wind_dir": ("wd", "wind_direction"),
        "v_sigma": ("sigmav", "v_sd", "sigma_v"),
        "ustar": ("u*",),
        "umean": ("ws", "ws_f", "wind_speed"),
        "mo_length": ("ol", "l"),
        "pblh": ("blh",),
    },
    "drivers": {
        "H": ("H", "H_F"),
        "TA": ("TA", "TA_F"),
        "PA": ("PA", "PA_F"),
        # Height metadata for zm = z - d (aerodynamic height).
        "measurement_height": ("measurement_height", "sensor_height",
                               "instrument_height"),
        "canopy_height": ("canopy_height", "vegetation_height", "hc"),
        "displacement": ("displacement", "displacement_height", "zd"),
    },
}

#: Driver keys matched case-sensitively (see :data:`ALIASES`).
_CASE_SENSITIVE_DRIVERS = frozenset({"H"})


class _ProcessedInputs(dict):
    """Processed model inputs, with estimation metadata out-of-band.

    Behaves exactly like the plain dict it used to be (so e.g.
    ``pd.DataFrame(inputs)`` keeps working); ``.estimated`` lists the names of
    the variables that were filled by an estimator.
    """

    def __init__(self, *args, estimated=(), **kwargs):
        super().__init__(*args, **kwargs)
        self.estimated = list(estimated)


def process_footprint_inputs(data=None, keep_cols=[], estimate_missing_variables=True,
                             fill_all=False, **kwargs):
    """
    Process input values for footprint calculation.

    Missing variables are estimated from the available ones when
    ``estimate_missing_variables`` is enabled (physically grounded estimates
    only). Crude constant fallbacks (e.g. ``wind_dir=0``, ``zm=30``,
    ``pblh=1000``) fabricate scientifically meaningful inputs, so they require
    an explicit ``fill_all=True`` opt-in; each one emits a ``UserWarning``.
    The names of every estimated variable are available on the returned
    mapping's ``.estimated`` attribute.

    Parameters:
        data (pd.DataFrame, optional): A DataFrame containing the required columns.
        **kwargs: Individual keyword arguments for the required values.

    Returns:
        dict: A dictionary with the processed input values as lists.
    """
    # Define the required keys
    required_keys = ['zm', 'z0', 'umean', 'ustar',
                     'pblh', 'mo_length', 'v_sigma', 'wind_dir'] + keep_cols
    aka_keys = ALIASES["model_inputs"]
    core_keys = ['zm', 'wind_dir']
    optional_keys = ['z0', 'umean'] + keep_cols
    # optional_keys = [] + keep_cols


    # If data is provided, extract values from the DataFrame
    if data is not None and isinstance(data, pd.DataFrame):
        # Drop full nan columns
        data = data.dropna(axis=1, how='all')

        if not isinstance(data, pd.DataFrame):
            raise ValueError("`data` must be a pandas DataFrame.")

        # Use regex to match column names case-insensitively
        inputs = {}
        for key in required_keys:
            # Create a regex pattern to match the key case-insensitively
            pattern = re.compile(f'^{re.escape(key)}$', re.IGNORECASE)
            # Find matching columns in the DataFrame, prioritizing exact matches
            matching_columns = [col for col in data.columns if col == key] + [
                col for col in data.columns if pattern.match(col)]
            
            if matching_columns:
                logger.debug(f'matching_columns: {matching_columns}')
                # Use the first matching column
                inputs[key] = data[matching_columns[0]].tolist()

        # Use other names variables may be known for (e.g. wind direction, wind_dir, wd)
        for key in required_keys:
            for aka in aka_keys.get(key, []):
                # Escape the alias: 'u*' must match the literal column name,
                # not act as a quantifier (which would also swallow a plain
                # 'U'/'u' wind-component column as friction velocity).
                pattern = re.compile(f'^{re.escape(aka)}$', re.IGNORECASE)
                # Find matching columns in the DataFrame, prioritizing exact matches
                matching_columns = [col for col in data.columns if col == key] + [
                    col for col in data.columns if pattern.match(col)]

                # if aka in data.columns:
                if matching_columns:
                    logger.debug(f'aka: {matching_columns[0]}')
                    inputs[key] = data[matching_columns[0]]

        # Check if the key is provided as a keyword argument
        for key in required_keys:
            if key in kwargs:
                logger.debug(f'kwargs {key}')
                inputs[key] = kwargs[key]

        # Met drivers consumed only by the estimators (e.g. the Obukhov length
        # from USTAR/H/TA/PA in FLUXNET/ICOS tables); never forwarded to the
        # models. 'H' (sensible heat flux, W m-2) is matched case-sensitively
        # so a lowercase 'h' column (FFP's name for the boundary-layer height)
        # is never mistaken for it. TA is degC, PA is kPa.
        for key, aliases in ALIASES["drivers"].items():
            exact = key in _CASE_SENSITIVE_DRIVERS
            # An explicit None kwarg means "not available": fall through to
            # the column scan instead of feeding None to the estimators.
            if kwargs.get(key) is not None:
                inputs[key] = kwargs[key]
                continue
            for name in aliases:
                if exact:
                    matching_columns = [c for c in data.columns if c == name]
                else:
                    pattern = re.compile(f'^{re.escape(name)}$', re.IGNORECASE)
                    matching_columns = [c for c in data.columns
                                        if pattern.match(c)]
                if matching_columns:
                    inputs[key] = data[matching_columns[0]].tolist()
                    break

    elif data is not None and isinstance(data, dict):
        # If DataFrame provided is a dict like kwargs
        data = {k: v for k, v in data.items() if v not in (None, '', [], {}, ())}

        inputs = data
        inputs.update(kwargs)
    elif data is not None:
        # Previously any other container fell through to kwargs-only inputs and
        # failed later with a confusing "missing inputs" error (or silently
        # computed from kwargs alone).
        raise TypeError(
            f"Unsupported `data` container: {type(data).__name__}. Supported: "
            "a pandas DataFrame or a dict of equal-length sequences "
            "(calculate_footprint additionally accepts a URL string). For an "
            "xarray.Dataset, pass dict(ds.data_vars) or ds.to_dataframe().")
    else:
        # If no data is provided, use kwargs
        inputs = kwargs

    # Estimate missing inputs when enabled, in dependency order (mo_length
    # before z0: the z0 estimate needs a real Obukhov length). `fill_all` also
    # allows crude constant fallbacks (zm, umean, wind_dir, pblh) and the rough
    # ustar estimate; with fill_all=False only physical estimates are applied.
    estimated = []
    if estimate_missing_variables:
        for key in ('zm', 'umean', 'wind_dir', 'ustar', 'mo_length', 'pblh',
                    'z0', 'v_sigma'):
            if inputs.get(key) is not None:
                continue
            value = micrometeorology.filler(inputs, key, fill_all=fill_all)
            if value is not None:
                inputs[key] = value
                estimated.append(key)

    # Core inputs are mandatory (after any estimation), and the models need
    # either z0 or umean (z0 takes precedence when both are present).
    missing_keys = [key for key in core_keys if inputs.get(key) is None]
    if inputs.get('z0') is None and inputs.get('umean') is None:
        missing_keys.append('z0 or umean')
    if missing_keys:
        raise ValueError(
            f"Missing required inputs: {missing_keys}. Provide them, or enable "
            f"approximation (estimate_missing_variables=True; set fill_all=True "
            f"for crude constant fallbacks).")

    # Remaining required (non-optional) inputs must also be present.
    missing_keys = [key for key in required_keys
                    if key not in optional_keys and inputs.get(key) is None]
    if missing_keys:
        raise ValueError(
            f"Missing required inputs: {missing_keys}. Provide them, or enable "
            f"approximation (estimate_missing_variables=True; set fill_all=True "
            f"for crude constant fallbacks).")
    
    # Get the maximum length of the inputs
    max_len_inputs = max(len(v) if isinstance(
        v, (list, np.ndarray)) else 1 for v in inputs.values())

    # Ensure all values are lists
    for key in required_keys:
        if key not in inputs:
            continue
        value = inputs[key]
        if isinstance(value, pd.Series):
            inputs[key] = value.tolist()
        elif not isinstance(value, (list, np.ndarray)):
            inputs[key] = [value]*max_len_inputs

    logger.debug(f'inputs: {inputs}')
    return _ProcessedInputs(inputs, estimated=estimated)


def _resolve_model(model):
    """Resolve a model name, callable, or module to a FootprintModel callable."""
    if isinstance(model, str):
        return get_model(model)
    if callable(model):
        return model
    calc = getattr(model, "calc", None)  # backwards compat: a model *module*
    if callable(calc):
        return calc
    raise TypeError(
        "`model` must be a registered model name, a FootprintModel callable, or "
        f"a module exposing calc(...); got {type(model).__name__}.")


def _group_label(key):
    """Map a groupby key to ``(time, label)`` for a Footprint.

    A timestamp becomes ``time``; a number becomes a relative ``time`` label
    (valid only in the local frame); anything else is kept as a ``group`` label
    with ``time=None``.
    """
    if key is None:
        return None, None
    if isinstance(key, (pd.Timestamp, datetime)):
        return pd.Timestamp(key).to_pydatetime(), key
    if isinstance(key, numbers.Number) and not isinstance(key, bool):
        return float(key), key
    return None, key


#: Inputs forwarded to a model's ``calc`` (met. variables + grid options).
_MODEL_KEYS = frozenset({
    "zm", "z0", "umean", "ustar", "pblh", "mo_length", "v_sigma", "wind_dir",
    "domain", "dx", "dy", "nx", "ny", "rslayer", "smooth_data", "verbosity",
})


def calculate_footprint(data=None, by=None, model="kljun2015", query=None,
                        tower=None, tower_crs=None, **kwargs):
    """Compute footprints from tabular inputs as a :class:`FootprintSeries`.

    Rows are grouped by ``by`` (one composited footprint per group) and each
    group is passed to the selected model. With ``by=None`` the whole table is a
    single group, giving a length-1 series. Use
    :meth:`FootprintSeries.aggregate` for a climatology, or index the series for
    individual footprints.

    Per-member smoothing defaults to off here (``smooth_data=0``) because
    :meth:`FootprintSeries.aggregate` smooths the climatology once, matching
    the reference FFP procedure; direct model calls keep their own default.
    Any estimated inputs are recorded in each footprint's
    ``attrs["estimated_inputs"]``.

    Args:
        data: A DataFrame, a dict of equal-length sequences, or a URL string.
        by: Column name (or list of names) to group rows by; ``None`` for one group.
        model: Registered model name (e.g. ``"kljun2015"``) or a FootprintModel
            callable. A module exposing ``calc(...)`` is also accepted.
        query: Optional pandas query applied to ``data`` before grouping.
        tower: ``(x, y)`` tower position, attached to each footprint for
            later georeferencing.
        tower_crs: CRS of ``tower``.
        **kwargs: Model inputs / grid options (e.g. ``domain``, ``dx``, ``zm``)
            and per-call overrides.

    Returns:
        FootprintSeries: One footprint per group, in the local tower-centred frame.

    Raises:
        ValueError: If no footprint could be calculated from the data.
    """
    model_fn = _resolve_model(model)

    if isinstance(data, str):
        data = io.read_from_url(data, na_values=[-9999])
    if isinstance(data, pd.DataFrame):
        data = data.copy()
    if query:
        data = data.query(query)

    keep_cols = (list(by) if isinstance(by, (list, tuple))
                 else [by] if isinstance(by, str) else [])
    inputs = process_footprint_inputs(data=data, keep_cols=keep_cols, **kwargs)
    estimated = getattr(inputs, "estimated", [])

    grouped = ([(None, inputs)] if by is None
               else pd.DataFrame(inputs).groupby(by))

    # Per-member smoothing is off in the batch path: the reference FFP
    # procedure smooths the *climatology* once (see FootprintSeries.aggregate),
    # and smoothing both the members and the aggregate widens the footprint.
    # Pass smooth_data=1 explicitly to override.
    grid_defaults = {"domain": [-500, 500, -500, 500], "dx": 10,
                     "smooth_data": 0, "verbosity": 0}
    overrides = {k: v for k, v in kwargs.items() if k in _MODEL_KEYS}

    footprints: list[Footprint] = []
    skipped = 0
    for key, group in grouped:
        if isinstance(group, pd.DataFrame):
            group = group.to_dict(orient="list")
        time, label = _group_label(key)
        call = {**grid_defaults, **overrides,
                **{k: v for k, v in group.items() if k in _MODEL_KEYS}}
        try:
            fp = model_fn(tower=tower, tower_crs=tower_crs, time=time, **call)
        except Exception as exc:
            # A failed group is skipped, never silently backfilled with another
            # group's footprint. Re-raise instead of `continue` to abort instead.
            logger.debug("Traceback for failed group %r:", key, exc_info=True)
            logger.warning("Footprint failed for group %r (%s); skipping.",
                           key, exc)
            skipped += 1
            continue
        if getattr(fp, "n", None) == 0:
            logger.warning("Group %r had no valid records; skipping.", key)
            skipped += 1
            continue
        if label is not None:
            if isinstance(label, (pd.Timestamp, datetime)):
                # Keep attrs NetCDF-serializable: a raw Timestamp in attrs
                # makes to_netcdf fail on the documented by-timestamp workflow.
                label = pd.Timestamp(label).isoformat()
            fp.attrs.setdefault("group", label)
        if estimated:
            fp.attrs.setdefault("estimated_inputs", list(estimated))
        footprints.append(fp)

    if not footprints:
        raise ValueError(
            "No footprints could be calculated from the provided data.")
    if skipped:
        logger.warning("Skipped %d of %d group(s).",
                       skipped, skipped + len(footprints))
    return FootprintSeries(footprints)


def empty_footprint(model="kljun2015", *, domain=None, dx=None, dy=None,
                    **kwargs) -> Footprint:
    """Return an empty (NaN) Footprint matching the model's grid.

    Runs the selected model once with placeholder inputs to obtain the exact
    grid it would produce for ``domain``/``dx``/``dy``, then blanks the field.
    Useful as a template (to pre-allocate, or to report the output shape)
    without computing a real footprint. Call ``.to_xarray()`` on the result for
    an empty DataArray/Dataset.

    Args:
        model: Registered model name or a FootprintModel callable.
        domain: ``[xmin, xmax, ymin, ymax]`` in metres (model default if None).
        dx, dy: Grid spacing in metres.
        **kwargs: Other grid options forwarded to the model (e.g. ``nx``/``ny``).

    Returns:
        Footprint: The grid/coords of a real footprint, with ``f`` all NaN.
    """
    model_fn = _resolve_model(model)
    grid = {"domain": domain if domain is not None else [-500, 500, -500, 500],
            "dx": dx if dx is not None else 10, "verbosity": 0}
    if dy is not None:
        grid["dy"] = dy
    grid.update({k: v for k, v in kwargs.items() if k in _MODEL_KEYS})

    # Safe placeholder met inputs: the grid is independent of their values, and
    # these avoid tripping the model's input validation. Skip any already given.
    placeholders = {"zm": 2.0, "umean": 2.0, "ustar": 0.3, "pblh": 1000.0,
                    "mo_length": -100.0, "v_sigma": 0.5, "wind_dir": 0.0}
    placeholders = {k: v for k, v in placeholders.items() if k not in grid}

    fp = model_fn(**grid, **placeholders)
    return fp._replace(f=np.full(fp.f.shape, np.nan, dtype=fp.f.dtype))


def wrapper(*args, out_as="nc", dst="", meta=None, aggregate=True,
            decimals=None, **kwargs):
    """Compute footprints and optionally write them to a file.

    A thin convenience over :func:`calculate_footprint`. Returns the climatology
    (``aggregate=True`` -> a :class:`Footprint`) or the full
    :class:`FootprintSeries`, and writes it to ``dst`` when given.

    Args:
        out_as: ``"nc"``/``"netcdf"`` or ``"tif"``/``"tiff"``. A TIFF needs an
            aggregated, georeferenced footprint.
        dst: Output path; nothing is written when empty.
        meta: Extra attributes merged into the result's ``attrs``.
        aggregate: Collapse the series to a climatology before returning/writing.
        *args, **kwargs: Forwarded to :func:`calculate_footprint`.

    Returns:
        Footprint | FootprintSeries: The computed result.
    """
    series = calculate_footprint(*args, **kwargs)
    result = series.aggregate() if aggregate else series

    # Metadata lives on Footprints (a series has no attrs of its own). Resolve
    # a callable/module model to a name so attrs stay NetCDF-serializable.
    model = kwargs.get("model", "kljun2015")
    model_name = model if isinstance(model, str) else \
        getattr(model, "__name__", type(model).__name__)
    targets = result.footprints if isinstance(result, FootprintSeries) else [result]
    for fp in targets:
        fp.attrs.setdefault("model", model_name)
        if meta:
            fp.attrs.update(meta)

    if dst:
        if out_as in ("nc", "netcdf"):
            result.to_netcdf(dst, decimals=decimals)
        elif out_as in ("tif", "tiff", "raster"):
            if isinstance(result, FootprintSeries):
                raise ValueError(
                    "Writing a TIFF needs a single footprint; pass "
                    "aggregate=True or index the series.")
            if not result.is_georeferenced:
                raise ValueError(
                    "Georeference the footprint before writing a TIFF: pass "
                    "tower/tower_crs and call .georeference(target_crs).")
            result.to_tiff(dst)
        else:
            raise ValueError(f"Unknown out_as={out_as!r}; use 'nc' or 'tif'.")
    return result


def aggregate_footprints(fclim_2d, dx, dy, smooth_data=1):
    """
    Aggregate multiple footprints into a single climatological footprint.

    .. deprecated:: 0.3
        Use :meth:`FootprintSeries.aggregate` instead; this raw-array helper
        will be removed in a future release.

    Parameters:
        footprints (list): List of footprint dictionaries.

    Returns:
        np.ndarray: Aggregated footprint.
    """
    warnings.warn(
        "fluxprint.aggregate_footprints is deprecated and will be removed in "
        "a future release; use FootprintSeries.aggregate() instead.",
        DeprecationWarning, stacklevel=2)
    fclim_2d = np.array(fclim_2d)
    if len(fclim_2d.shape) == 2:
        logger.info(
            f"Footprint must be 3D (time, x, y), dimension passed was: {fclim_2d.shape}.")
        return fclim_2d

    assert len(
        fclim_2d.shape) == 3, f"Footprint must be 3D (time, x, y), dimension passed was: {fclim_2d.shape}."
    #n_valid = len(fclim_2d)

    fclim_clim = np.nanmean(fclim_2d, axis=0)

    # Truthiness, not `is not None`: smooth_data=0 must disable smoothing.
    if smooth_data:
        fclim_clim = smooth_field(fclim_clim)
    return fclim_clim


def get_contour(footprint, dx, dy, rs, verbosity=0):
    """Source-area contours of a legacy footprint object.

    .. deprecated:: 0.3
        Use :meth:`Footprint.contours` instead; this helper will be removed
        in a future release. A :class:`Footprint` argument is delegated there.
    """
    warnings.warn(
        "fluxprint.get_contour is deprecated and will be removed in a future "
        "release; use Footprint.contours() instead.",
        DeprecationWarning, stacklevel=2)
    if isinstance(footprint, Footprint):
        return footprint.contours(rs if rs is not None else (0.5, 0.8))
    if rs is None:
        raise ValueError(
            "rs is required for the legacy get_contour path (e.g. "
            "rs=[0.5, 0.8]).")
    from . import utils  # deferred: pulls the heavy geo/plotting stack
    flag_err = 0

    footprint = utils.convert_to_object(
        footprint)

    # Handle rs
    if rs is not None:

        # Check that rs is a list, otherwise make it a list
        if isinstance(rs, numbers.Number):
            if 0.9 < rs <= 1 or 90 < rs <= 100:
                rs = 0.9
            rs = [rs]
        if not isinstance(rs, list):
            exceptions.raise_ffp_exception(18, verbosity)

        # If rs is passed as percentages, normalize to fractions of one
        if np.max(rs) >= 1:
            rs = [x/100. for x in rs]

        # Eliminate any values beyond 0.9 (90%) and inform user
        if np.max(rs) > 0.9:
            exceptions.raise_ffp_exception(19, verbosity)
            rs = [item for item in rs if item <= 0.9]

        # Sort levels in ascending order
        rs = list(np.sort(rs))

    # Derive footprint ellipsoid incorporating R% of the flux, if requested,
    # starting at peak value.
    if rs is not None:
        clevs = utils.get_contour_levels(footprint.fclim_2d, dx, dy, rs)
        frs = [item[2] for item in clevs]
        xrs = []
        yrs = []
        for ix, fr in enumerate(frs):
            xr, yr = utils.get_contour_vertices(
                footprint.x_2d, footprint.y_2d, footprint.fclim_2d, fr)
            if xr is None:
                frs[ix] = None
                flag_err = 2
            xrs.append(xr)
            yrs.append(yr)

    # footprint.update({"xr": xrs, "yr": yrs, 'fr': frs, 'rs': rs})
    # return footprint
    return type('var_', (object,), {"xr": xrs, "yr": yrs, 'fr': frs, 'rs': rs, 'flag_err': flag_err})


__all__ = [
    "ALIASES",
    "calculate_footprint",
    "empty_footprint",
    "process_footprint_inputs",
    "aggregate_footprints",
    "get_contour",
    "wrapper",
]
