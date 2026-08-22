"""Estimators for missing micrometeorological inputs.

Approximations come in two tiers:

* **essential** - physically grounded estimates derived from other inputs
  (``z0``, ``mo_length``, ``v_sigma``);
* **filler** - crude constant fallbacks (``zm``, ``umean``, ``wind_dir``,
  ``pblh``) and a rough ``ustar``, applied only when ``fill_all`` is enabled.
  Every constant fallback emits a :class:`UserWarning` naming the fabricated
  value.

:func:`filler` returns an estimated value for a variable, or ``None`` when no
estimator applies, an estimator's inputs are unavailable, or its tier is
disabled. This lets callers allow or disable approximation when inputs are
missing.
"""
from __future__ import annotations

import logging
import warnings
from collections.abc import Mapping
from typing import Any

import numpy as np
from regorator import create_registry, register

logger = logging.getLogger("fluxprint.micrometeorology")

__all__ = ["filler", "caller", "ESTIMATORS"]

#: Registry of physically grounded estimators (name -> callable).
ESTIMATORS = create_registry("fluxprint micrometeorological estimators")


@register("z0", ESTIMATORS, "Roughness length from umean, ustar, zm, mo_length")
def compute_z0(umean, ustar, zm, psi_f=None, ol=None, k=0.4):
    """From Kljun.py (not yet validated)."""
    if psi_f is None:
        psi_f = compute_psi_f(zm, ol)
    exponent = (np.asarray(umean) / np.asarray(ustar)) * k + psi_f
    return np.asarray(zm) / np.exp(exponent)


@register("psi_f", ESTIMATORS, "Stability correction function for momentum")
def compute_psi_f(zm, ol):
    """From Kljun.py."""
    oln = 5000  # L limit for neutral scaling
    zm, ol = np.asarray(zm), np.asarray(ol)
    xx = (1 - 19.0 * zm / ol) ** 0.25
    psi_f = np.zeros_like(xx) * np.nan
    psi_f = np.where(
        (ol <= 0) | (ol >= oln),
        np.log((1 + xx**2) / 2.) + 2. * np.log((1 + xx) / 2.)
        - 2. * np.arctan(xx) + np.pi / 2, psi_f)
    psi_f = np.where((ol > 0) & (ol < oln), -5.3 * zm / ol, psi_f)
    return psi_f


@register("pblh", ESTIMATORS, "Boundary-layer height from ustar and latitude")
def compute_pblh(ustar, latitude_deg, f_min=1e-5):
    """Boundary-layer height ``h = ustar / f``.

    The Coriolis parameter f -> 0 near the equator, which would blow up
    ``h = ustar / f``, so ``|f|`` is floored at ``f_min`` (~ +/- 4 deg
    latitude) while keeping its sign.
    """
    omega = 7.2921e-5  # Earth's angular velocity [rad s-1]
    f = 2 * omega * np.sin(np.radians(latitude_deg))
    f = np.where(f >= 0, np.maximum(f, f_min), np.minimum(f, -f_min))
    return np.asarray(ustar) / f


def compute_virtual_potential_temperature(Ta, P, r=None, P0=100, R_cp=0.286, r_L=0):
    """Not yet validated."""
    theta = np.asarray(Ta) * (P0 / np.asarray(P)) ** R_cp
    if r is not None:
        return theta * (1 + 0.61 * r - r_L)
    return theta


@register("mo_length", ESTIMATORS, "Obukhov length from ustar, theta and heat flux")
def compute_mo_length(ustar, H, theta=None, TA=None, PA=None, k=0.4, g=9.81,
                      cp=1005.0, Rd=287.05):
    """Obukhov length from ustar, sensible heat flux and temperature.

    ``H`` is the *sensible* heat flux [W m-2], ``TA`` is in degrees Celsius and
    ``PA`` in kPa. The Obukhov length is defined with the *kinematic* buoyancy
    flux ``w'theta' = H / (rho*cp)`` [K m s-1], so ``H`` must be divided by
    ``rho*cp`` (~1200) first -- otherwise ``|L|`` comes out ~1200x too small.
    """
    if TA is None or PA is None:
        raise ValueError(
            "compute_mo_length requires TA (degC) and PA (kPa) to convert the "
            "sensible heat flux H (W m-2) into the kinematic buoyancy flux "
            "H/(rho*cp).")
    T_K = np.asarray(TA) + 273.15
    if theta is None:
        theta = compute_virtual_potential_temperature(T_K, PA)
    # Air density [kg m-3] from the ideal gas law (PA kPa -> Pa).
    rho = (np.asarray(PA) * 1000.0) / (Rd * T_K)
    # Kinematic (buoyancy) heat flux w'theta' [K m s-1].
    w_theta = np.asarray(H) / (rho * cp)
    with np.errstate(divide="ignore"):
        mo_length = -(np.asarray(ustar) ** 3) * theta / (k * g * w_theta)
    # H = 0 (neutral) gives |L| -> inf; clamp to a large finite value so the
    # record is treated as the neutral limit (|L| > 5000 in the models) rather
    # than rejected as non-finite.
    return np.clip(mo_length, -1e6, 1e6)


@register("v_sigma", ESTIMATORS, "Std. dev. of lateral velocity from ustar")
def compute_std_v(ustar, a=2.0, b=0):
    """Crude sigma_v ~ a * ustar (a ~ 1.9-2.5 near-neutral; was a=3.5)."""
    return a * np.asarray(ustar) + b


@register("ustar", ESTIMATORS, "Friction velocity from umean, zm and z0")
def compute_ustar(umean, zm, z0=0.1, k=0.4):
    """Not yet validated."""
    return (np.asarray(umean) * k) / np.log(np.asarray(zm) / z0)


@register("displacement", ESTIMATORS, "Displacement height from canopy height")
def compute_displacement(canopy_height, c=0.67):
    """Zero-plane displacement ``d ~ c * canopy_height`` (rule of thumb).

    The models' ``zm`` is the *aerodynamic* measurement height ``z - d``, not
    the instrument height; over vegetation the difference is substantial.
    """
    return c * np.asarray(canopy_height)


def _zm_from_heights(d):
    """zm = measurement height - displacement (from d or canopy height)."""
    displacement = d.get("displacement")
    if displacement is None:
        if d.get("canopy_height") is None:
            return None
        displacement = compute_displacement(d["canopy_height"])
    return np.asarray(d["measurement_height"]) - np.asarray(displacement)


def _essential() -> dict[str, tuple]:
    # variable -> (constant value | callable(data), required input keys)
    return {
        # zm from measurement height minus displacement (z - d): the models
        # need the aerodynamic height, which flux metadata rarely lists.
        "zm": (_zm_from_heights, ("measurement_height",)),
        # z0 requires a real Obukhov length: a defaulted ol here (formerly
        # ol=1 m, extreme stability) made psi_f huge and z0 astronomically
        # wrong. Estimation order in core computes mo_length before z0.
        "z0": (lambda d: compute_z0(d["umean"], d["ustar"], d["zm"],
                                    ol=d["mo_length"]),
               ("umean", "ustar", "zm", "mo_length")),
        "mo_length": (lambda d: compute_mo_length(d["ustar"], d["H"],
                                                  TA=d["TA"], PA=d["PA"]),
                      ("ustar", "H", "TA", "PA")),
        "v_sigma": (lambda d: compute_std_v(d["ustar"]), ("ustar",)),
    }


def _filler() -> dict[str, tuple]:
    return {
        "zm": (30.0, ()),
        "umean": (1.0, ()),
        "ustar": (lambda d: compute_ustar(d["umean"], d["zm"], z0=d.get("z0", 0.1)),
                  ("umean", "zm")),
        "wind_dir": (0.0, ()),
        # A constant boundary-layer height is a crude fallback, not a physical
        # estimate; it must be opted into via fill_all like the others.
        "pblh": (1000.0, ()),
    }


def filler(data: Mapping[str, Any], variable: str, fill_all: bool = True):
    """Estimate a missing variable, or return ``None`` if it can't be filled.

    Args:
        data: Mapping of inputs already available.
        variable: Name of the variable to estimate.
        fill_all: Also allow crude constant fallbacks (``zm``/``umean``/
            ``wind_dir``/``pblh``) and the rough ``ustar`` estimate. With
            ``False`` only physically grounded ("essential") estimates are used.

    Returns:
        The estimated value, or ``None`` when unavailable/disabled.
    """
    tables = [(_essential(), False)]
    if fill_all:
        tables.append((_filler(), True))

    for table, crude in tables:
        entry = table.get(variable)
        if entry is None:
            continue
        spec, needs = entry
        # Present-but-None counts as unavailable (callers commonly pass None
        # for absent variables); dereferencing it would crash the estimator.
        if any(data.get(key) is None for key in needs):
            logger.debug("Cannot estimate %r: missing inputs %s.",
                         variable, [k for k in needs if data.get(k) is None])
            continue
        value = spec(data) if callable(spec) else spec
        if value is None:  # estimator declined (e.g. optional inputs absent)
            continue
        if crude and not callable(spec):
            message = f"Using crude fallback for missing {variable!r}: {spec}"
            logger.warning("%s", message)
            warnings.warn(message, UserWarning, stacklevel=2)
        return value
    return None


def caller(data: Mapping[str, Any], variable: str):
    """Backwards-compatible alias for :func:`filler` (``fill_all=True``)."""
    return filler(data, variable, fill_all=True)
