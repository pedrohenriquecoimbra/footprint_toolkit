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

from ..exceptions import raise_ffp_exception

logger = logging.getLogger('fluxprint.model.engine')

__all__ = ["NormalizedInputs", "listify", "normalize_inputs"]


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
