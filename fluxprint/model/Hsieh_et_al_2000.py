"""Hsieh et al. (2000) analytical footprint model.

Registered as ``"hsieh2000"``: the crosswind-integrated footprint of Hsieh,
Katul & Chi (2000, Adv. Water Resour. 23, 765-772), expanded to two
dimensions with the Gaussian crosswind dispersion of Detto et al. (2006,
Water Resour. Res. 42, W08419, Appendix B).

The model is fully analytic, so the kernel evaluates the wind-aligned
footprint directly at the wind-frame coordinates of the fixed output grid
(:func:`fluxprint.grid.to_wind_frame`) — the grid never rotates, and the
returned field is a density in m**-2 whose full integral is one (the
:class:`~fluxprint.footprint.Footprint` contract). Everything except the
equations below (the grid, the record loop, validation plumbing, smoothing,
provenance) comes from the generic driver via
:func:`~fluxprint.model.engine.footprint_model`.

Differences from the pre-0.4 experimental draft (which was never registered):
the draft rotated the *grid* into the wind (producing an irregular grid) with
a math-angle rotation instead of the meteorological azimuth, interpolated
from a native grid, and returned per-pixel fractions instead of a density.
All three are corrected here; the physics equations are unchanged.

Inputs follow the canonical model signature. ``z0`` is required (the model
has no ``umean`` mode — records without ``z0`` are rejected), and ``pblh``
participates only in input validation (``zm > pblh`` is rejected), not in
the footprint equations.
"""
import numpy as np

from ..grid import to_wind_frame
from .engine import ffp_validate, footprint_model

__all__ = ["calc", "peak_distance"]

#: von Karman constant (as in the reference implementations).
_K = 0.4

#: Similarity coefficients (D, P) by stability class (Hsieh 2000, Table 2 /
#: Eq 17): unstable, near neutral, stable — classified on zu/L with the
#: paper's 0.04 threshold.
_STABILITY = {
    "unstable": (0.28, 0.59),
    "neutral": (0.97, 1.0),
    "stable": (2.44, 1.33),
}
_STAB_THRESHOLD = 0.04


def _hsieh_params(zm, z0, mo_length):
    """``(zu, D, P, A)`` for one record.

    ``zu`` is the height scale of Hsieh Eq 13.5; ``A = D zu**P |L|**(1-P) /
    k**2`` is the along-wind length scale that carries the whole
    crosswind-integrated footprint: ``f_ci(x) = (A / x**2) exp(-A / x)``,
    cumulative ``F(x) = exp(-A / x)``, peak at ``A / 2`` (Eq 19).
    """
    zu = zm * (np.log(zm / z0) - 1 + z0 / zm)
    stab = zu / mo_length
    if stab < -_STAB_THRESHOLD:
        D, P = _STABILITY["unstable"]
    elif stab <= _STAB_THRESHOLD:
        D, P = _STABILITY["neutral"]
    else:
        D, P = _STABILITY["stable"]
    A = D * zu**P * abs(mo_length) ** (1 - P) / _K**2
    return zu, D, P, A


def peak_distance(*, zm, z0, mo_length):
    """Upwind distance of the footprint maximum [m] (Hsieh 2000, Eq 19).

    The most common single number asked of a footprint model — the fetch of
    peak contribution — available without computing a field.
    """
    return _hsieh_params(zm, z0, mo_length)[3] / 2.0


def _hsieh_validate(rec, opts, verbosity, *, quiet=False):
    """Hsieh requires ``z0`` (no ``umean`` mode), plus the standard checks."""
    if rec.z0 is None:
        return False
    return ffp_validate(rec, opts, verbosity, quiet=quiet)


def _hsieh_record(ctx, rec, opts):
    """Per-record Hsieh/Detto footprint density on the fixed grid.

    ``f(x_along, x_cross) = f_ci(x_along) * N(x_cross; sigma_y(x_along))``
    with ``f_ci`` from Hsieh Eq 17 and ``sigma_y = 0.3 z0 (sigma_v / ustar)
    (x / z0)**0.86`` from Detto Eq B4, evaluated at the wind-frame
    coordinates of the output grid (along-wind positive toward the wind
    source, i.e. the fetch).
    """
    _zu, _D, _P, A = _hsieh_params(rec.zm, rec.z0, rec.mo_length)

    along, cross = to_wind_frame(ctx.x_2d, ctx.y_2d, rec.wind_dir)
    f_2d = np.zeros(ctx.x_2d.shape)
    px = along > 0
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        t = A / along[px]
        f_ci = (t / along[px]) * np.exp(-t)
        sy = 0.3 * rec.z0 * (rec.v_sigma / rec.ustar) \
            * (along[px] / rec.z0) ** 0.86
        f_2d[px] = (f_ci / (np.sqrt(2 * np.pi) * sy)
                    * np.exp(-cross[px] ** 2 / (2 * sy**2)))
    # Numerical underflow at the near-field limit (x -> 0+: f -> 0, but the
    # intermediate terms can overflow) must land at the analytic limit, 0.
    f_2d[~np.isfinite(f_2d)] = 0.0
    return f_2d, 0, 1


#: Provenance stamped into every footprint this model produces.
HSIEH_META = {
    "model_citation": ("Hsieh, C.-I., G. Katul, T. Chi (2000): An "
                       "approximate analytical model for footprint "
                       "estimation of scalar fluxes in thermally stratified "
                       "atmospheric flows. Adv. Water Resour. 23, 765-772."),
    "model_doi": "10.1016/S0309-1708(99)00042-1",
    "lateral_citation": ("Detto, M., N. Montaldo, J.D. Albertson, M. "
                         "Mancini, G. Katul (2006), Water Resour. Res. 42, "
                         "W08419, Appendix B (crosswind dispersion)."),
}

calc = footprint_model(
    "hsieh2000",
    description=("Hsieh et al. (2000) analytical footprint with "
                 "Detto et al. (2006) crosswind expansion"),
    meta=HSIEH_META,
    validate=_hsieh_validate,
)(_hsieh_record)
