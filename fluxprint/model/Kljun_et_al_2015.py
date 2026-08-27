"""

"""

import logging
import warnings

import numpy as np


from ..exceptions import InputValidationError, exceptions, raise_ffp_exception
from ..grid import GridContext, resolve_grid
from . import engine

logger = logging.getLogger('fluxprint.model.kljun_et_al_2015')

# from __future__ import print_function


def _kljun_record(ctx, rec, opts):
    """Kljun et al. (2015) per-record footprint field on the fixed grid.

    The FFP reference's loop body, moved verbatim (bug-for-bug: this path is
    pinned bitwise to the vendored reference — do not clean anything up).
    Returns ``(f_2d, flag, valid)``; ``flag=3`` marks a record that proved
    invalid mid-computation (``log(zm/z0) <= psi_f``).
    """
    #===========================================================================
    # Model parameters
    a = 1.4524
    b = -1.9914
    c = 1.4622
    d = 0.1359
    ac = 2.17
    bc = 1.66
    cc = 20.0

    oln = 5000 #limit to L for neutral scaling
    k = 0.4 #von Karman

    ustar, v_sigma, pblh, mo_length, wind_dir, zm, z0, umean = rec
    x_2d = ctx.x_2d
    rho = ctx.rho
    theta = ctx.theta
    flag = 0
    valid = 1

    #===========================================================================
    # Rotate coordinates into wind direction
    if wind_dir is not None:
        rotated_theta = theta - wind_dir * np.pi / 180.

    #===========================================================================
    # Create real scale crosswind integrated footprint and dummy for
    # rotated scaled footprint
    fstar_ci_dummy = np.zeros(x_2d.shape)
    f_ci_dummy = np.zeros(x_2d.shape)
    xstar_ci_dummy = np.zeros(x_2d.shape)
    px = np.ones(x_2d.shape)
    if z0 is not None:
        # Use z0
        if mo_length <= 0 or mo_length >= oln:
            xx = (1 - 19.0 * zm/mo_length)**0.25
            psi_f = (np.log((1 + xx**2) / 2.) + 2. * np.log((1 + xx) / 2.) - 2. * np.arctan(xx) + np.pi/2)
        elif mo_length > 0 and mo_length < oln:
            psi_f = -5.3 * zm / mo_length
        if (np.log(zm / z0)-psi_f)>0:
            xstar_ci_dummy = (rho * np.cos(rotated_theta) / zm * (1. - (zm / pblh)) / (np.log(zm / z0) - psi_f))
            px = np.where(xstar_ci_dummy > d)
            fstar_ci_dummy[px] = a * (xstar_ci_dummy[px] - d)**b * np.exp(-c / (xstar_ci_dummy[px] - d))
            f_ci_dummy[px] = (fstar_ci_dummy[px] / zm * (1. - (zm / pblh)) / (np.log(zm / z0) - psi_f))
        else:
            flag = 3
            valid = 0
    else:
        # Use umean if z0 not available
        xstar_ci_dummy = (rho * np.cos(rotated_theta) / zm * (1. - (zm / pblh)) / (umean / ustar * k))
        px = np.where(xstar_ci_dummy > d)
        fstar_ci_dummy[px] = a * (xstar_ci_dummy[px] - d)**b * np.exp(-c / (xstar_ci_dummy[px] - d))
        f_ci_dummy[px] = (fstar_ci_dummy[px] / zm * (1. - (zm / pblh)) / (umean / ustar * k))

    #===========================================================================
    # Calculate dummy for scaled sig_y* and real scale sig_y
    sigystar_dummy = np.zeros(x_2d.shape)
    sigystar_dummy[px] = (ac * np.sqrt(bc * np.abs(xstar_ci_dummy[px])**2 / (1 +
                          cc * np.abs(xstar_ci_dummy[px]))))

    if abs(mo_length) > oln:
        mo_length = -1E6
    if mo_length <= 0:   #convective
        scale_const = 1E-5 * abs(zm / mo_length)**(-1) + 0.80
    elif mo_length > 0:  #stable
        scale_const = 1E-5 * abs(zm / mo_length)**(-1) + 0.55
    if scale_const > 1:
        scale_const = 1.0

    sigy_dummy = np.zeros(x_2d.shape)
    sigy_dummy[px] = (sigystar_dummy[px] / scale_const * zm * v_sigma / ustar)
    sigy_dummy[sigy_dummy < 0] = np.nan

    #===========================================================================
    # Calculate real scale f(x,y)
    f_2d = np.zeros(x_2d.shape)
    f_2d[px] = (f_ci_dummy[px] / (np.sqrt(2 * np.pi) * sigy_dummy[px]) *
                np.exp(-(rho[px] * np.sin(rotated_theta[px]))**2 / ( 2. * sigy_dummy[px]**2)))

    return f_2d, flag, valid


def calc_ffp_climatology(zm=None, z0=None, umean=None, pblh=None, mo_length=None, v_sigma=None, ustar=None,
                    wind_dir=None, domain=None, dx=None, dy=None, nx=None, ny=None,
                    rs=None, rslayer=0,
                    smooth_data=1, crop=False, pulse=None, verbosity=2, **kwargs):
    """
    Derive a flux footprint estimate based on the simple parameterisation FFP
    See Kljun, N., P. Calanca, M.W. Rotach, H.P. Schmid, 2015:
    The simple two-dimensional parameterisation for Flux Footprint Predictions FFP.
    Geosci. Model Dev. 8, 3695-3713, doi:10.5194/gmd-8-3695-2015, for details.
    contact: natascha.kljun@cec.lu.se

    This function calculates footprints within a fixed physical domain for a series of
    time steps, rotates footprints into the corresponding wind direction and aggregates
    all footprints to a footprint climatology. The percentage of source area is
    calculated for the footprint climatology.
    For determining the optimal extent of the domain (large enough to include footprints)
    use calc_footprint_FFP.py.

    FFP Input
        All vectors need to be of equal length (one value for each time step)
        zm       = Measurement height above displacement height (i.e. z-d) [m]
                   usually a scalar, but can also be a vector 
        z0       = Roughness length [m] - enter [None] if not known 
                   usually a scalar, but can also be a vector 
        umean    = Vector of mean wind speed at zm [ms-1] - enter [None] if not known 
                   Either z0 or umean is required. If both are given,
                   z0 is selected to calculate the footprint
        pblh        = Vector of boundary layer height [m]
        mo_length       = Vector of Obukhov length [m]
        v_sigma   = Vector of standard deviation of lateral velocity fluctuations [ms-1]
        ustar    = Vector of friction velocity [ms-1]
        wind_dir = Vector of wind direction in degrees (of 360) for rotation of the footprint     

        Optional input:
        domain       = Domain size as an array of [xmin xmax ymin ymax] [m].
                       Footprint will be calculated for a measurement at [0 0 zm] m
                       Default is smallest area including the r% footprint or [-1000 1000 -1000 1000]m,
                       whichever smallest (80% footprint if r not given).
        dx, dy       = Cell size of domain [m]
                       Small dx, dy results in higher spatial resolution and higher computing time
                       Default is dx = dy = 2 m. If only dx is given, dx=dy.
        nx, ny       = Two integer scalars defining the number of grid elements in x and y
                       Large nx/ny result in higher spatial resolution and higher computing time
                       Default is nx = ny = 1000. If only nx is given, nx=ny.
                       If both dx/dy and nx/ny are given, dx/dy is given priority if the domain is also specified.
        rs           = DEPRECATED, ignored: source areas moved to
                       Footprint.contours() / Footprint.level_for().
        rslayer      = Calculate footprint even if zm within roughness sublayer: set rslayer = 1
                       Note that this only gives a rough estimate of the footprint as the model is not
                       valid within the roughness sublayer. Default is 0 (i.e. no footprint for within RS).
                       z0 is needed for estimation of the RS.
        smooth_data  = Apply convolution filter to smooth footprint climatology if smooth_data=1 (default)
        crop         = DEPRECATED, ignored: crop via Footprint.contours() and
                       slicing instead.
        pulse        = Display progress of footprint calculations every pulse-th footprint (e.g., "100")
        verbosity    = Level of verbosity at run time: 0 = completely silent, 1 = notify only of fatal errors,
                       2 = all notifications

    FFP output
        FFP      = Structure array with footprint climatology data for measurement at [0 0 zm] m
        x_2d	    = x-grid of 2-dimensional footprint [m]
        y_2d	    = y-grid of 2-dimensional footprint [m]
        fclim_2d = Normalised footprint function values of footprint climatology [m-2]
        n        = Number of footprints calculated and included in footprint climatology
        flag_err = 0 if no error, 1 in case of error,
                   3 if single data points had to be removed (outside validity)

    Created: 19 May 2016 natascha kljun
    Converted from matlab to python, together with Gerardo Fratini, LI-COR Biosciences Inc.
    version: 1.42
    last change: 11/12/2019 Gerardo Fratini, ported to Python 3.x
    Copyright (C) 2015 - 2023 Natascha Kljun
    """

    if crop:
        warnings.warn(
            "calc_ffp_climatology(crop=...) is deprecated and ignored; crop "
            "via Footprint.contours()/level_for() and array slicing instead.",
            DeprecationWarning, stacklevel=2)
    if rs is not None:
        warnings.warn(
            "calc_ffp_climatology(rs=...) is deprecated and has never had an "
            "effect in this port; source areas moved to Footprint.contours().",
            DeprecationWarning, stacklevel=2)

    #===========================================================================
    # Input check (hoisted verbatim into fluxprint.model.engine)
    inputs = engine.normalize_inputs(
        zm=zm, ustar=ustar, pblh=pblh, mo_length=mo_length, v_sigma=v_sigma,
        wind_dir=wind_dir, z0=z0, umean=umean, verbosity=verbosity)

    # Define rslayer if not passed
    if rslayer is None: rslayer = 0

    # Define smooth_data if not passed
    if smooth_data is None: smooth_data = 1

    #===========================================================================
    # Computational domain (fluxprint.grid: the reference's reconciliation
    # rules, verbatim); the per-record physics is _kljun_record, driven by
    # the model-agnostic engine loop.
    ctx = GridContext(resolve_grid(domain=domain, dx=dx, dy=dy, nx=nx, ny=ny))
    result = engine.run_climatology(
        _kljun_record, ctx=ctx, inputs=inputs, opts={"rslayer": rslayer},
        validate=engine.ffp_validate, smooth_data=smooth_data, pulse=pulse,
        verbosity=verbosity)

    #===========================================================================
    # Fill output structure
    return type('var_', (object,), {'x_2d': ctx.x_2d, 'y_2d': ctx.y_2d,
                'fclim_2d': result.fclim_2d,
                'n': result.n, 'flag_err': result.flag_err})


def calc_footprint_1d(zm=None, z0=None, umean=None, pblh=None, mo_length=None, v_sigma=None, ustar=None,
                      nx=1000, **kwargs):
    """
    Derive a flux footprint estimate based on the simple parameterisation FFP
    See Kljun, N., P. Calanca, M.W. Rotach, H.P. Schmid, 2015: 
    The simple two-dimensional parameterisation for Flux Footprint Predictions FFP.
    Geosci. Model Dev. 8, 3695-3713, doi:10.5194/gmd-8-3695-2015, for details.
    contact: natascha.kljun@cec.lu.se

    FFP Input
    zm     = Measurement height above displacement height (i.e. z-d) [m]
    z0     = Roughness length [m]; enter None if not known 
    umean  = Mean wind speed at zm [m/s]; enter None if not known 
             Either z0 or umean is required. If both are given,
             z0 is selected to calculate the footprint
    pblh      = Boundary layer height [m]
    mo_length     = Obukhov length [m]
    v_sigma = standard deviation of lateral velocity fluctuations [ms-1]
	ustar  = friction velocity [ms-1]

    optional inputs:
    nx       = Integer scalar defining the number of grid elements of the scaled footprint.
               Large nx results in higher spatial resolution and higher computing time.
               Default is 1000, nx must be >=600.
 
    FFP output
    x_ci_max = x location of footprint peak (distance from measurement) [m]
    x_ci	 = x array of crosswind integrated footprint [m]
    f_ci	 = array with footprint function values of crosswind integrated footprint [m-1] 
    x_2d	 = x-grid of 2-dimensional footprint [m], rotated if wind_dir is provided
    y_2d	 = y-grid of 2-dimensional footprint [m], rotated if wind_dir is provided
    f_2d	 = footprint function values of 2-dimensional footprint [m-2]
    rs       = percentage of footprint as in input, if provided
    fr       = footprint value at r, if r is provided
    xr       = x-array for contour line of r, if r is provided
    yr       = y-array for contour line of r, if r is provided
    flag_err = 0 if no error, 1 in case of error

    created: 15 April 2015 natascha kljun
    translated to python, December 2015 Gerardo Fratini, LI-COR Biosciences Inc.
    version: 1.42
    last change: 11/12/2019 Gerardo Fratini, ported to Python 3.x
    Copyright (C) 2015 - 2023 Natascha Kljun
    """

    # ===========================================================================
    # Input check
    flag_err = 0

    def _reject(code):
        # The shared table classes these as non-fatal 'error' codes, but for a
        # single-record function a failed check must abort, not just log.
        ex = [it for it in exceptions if it['code'] == code][0]
        raise InputValidationError(
            f"{ex['type']}({str(code).zfill(4)}): {ex['msg']}", code=code)

    # Check existence of required input pars
    if None in [zm, pblh, mo_length, v_sigma, ustar] or (z0 is None and umean is None):
        raise_ffp_exception(1)

    # Check passed values (reject NaN/inf first: comparisons cannot catch it)
    if not all(np.isfinite(val) for val in
               [zm, pblh, mo_length, v_sigma, ustar,
                umean if z0 is None else z0]):
        _reject(21)
    if zm <= 0.:
        _reject(2)
    if z0 is not None and umean is None and z0 <= 0.:
        _reject(3)
    if pblh <= 10.:
        _reject(4)
    if zm > pblh:
        _reject(5)
    if z0 is not None and umean is None and zm <= 12.5*z0:
        _reject(20)
    # ol == 0 is unphysical and would ZeroDivisionError, same guard as
    # check_ffp_inputs.
    if mo_length == 0 or float(zm)/mo_length <= -15.5:
        _reject(7)
    if v_sigma <= 0:
        _reject(8)
    if ustar <= 0.1:
        _reject(9)
    if nx < 600:
        raise InputValidationError(
            'nx (number of grid elements) must be >= 600.')

    # Resolve ambiguity if both z0 and umean are passed (defaults to using z0)
    if None not in [z0, umean]:
        raise_ffp_exception(13)

    # ===========================================================================
    # Model parameters
    a = 1.4524
    b = -1.9914
    c = 1.4622
    d = 0.1359
    ac = 2.17
    bc = 1.66
    cc = 20.0

    xstar_end = 30
    oln = 5000  # limit to L for neutral scaling
    k = 0.4  # von Karman

    # ===========================================================================
    # Scaled X* for crosswind integrated footprint
    xstar_ci_param = np.linspace(d, xstar_end, nx+2)
    xstar_ci_param = xstar_ci_param[1:]

    # Crosswind integrated scaled F*
    fstar_ci_param = a * (xstar_ci_param-d)**b * \
        np.exp(-c / (xstar_ci_param-d))
    ind_notnan = ~np.isnan(fstar_ci_param)
    fstar_ci_param = fstar_ci_param[ind_notnan]
    xstar_ci_param = xstar_ci_param[ind_notnan]

    # Scaled sig_y*
    sigystar_param = ac * \
        np.sqrt(bc * xstar_ci_param**2 / (1 + cc * xstar_ci_param))

    # ===========================================================================
    # Real scale x and f_ci
    if z0 is not None:
        # Use z0
        if mo_length <= 0 or mo_length >= oln:
            xx = (1 - 19.0 * zm/mo_length)**0.25
            psi_f = np.log((1 + xx**2) / 2.) + 2. * \
                np.log((1 + xx) / 2.) - 2. * np.arctan(xx) + np.pi/2
        elif mo_length > 0 and mo_length < oln:
            psi_f = -5.3 * zm / mo_length

        x = xstar_ci_param * zm / (1. - (zm / pblh)) * (np.log(zm / z0) - psi_f)
        if np.log(zm / z0) - psi_f > 0:
            x_ci = x
            f_ci = fstar_ci_param / zm * \
                (1. - (zm / pblh)) / (np.log(zm / z0) - psi_f)
        else:
            # log(zm/z0) <= psi_f: no valid solution for this z0/zm/L combo.
            return {'x_ci_max': None, 'x_ci': None, 'f_ci': None,
                    'x': None, 'f': None, 'sigy': None, 'f_1d': None,
                    'fstar_ci_param': fstar_ci_param, 'flag_err': 1}
    else:
        # Use umean if z0 not available
        x = xstar_ci_param * zm / (1. - zm / pblh) * (umean / ustar * k)
        if umean / ustar > 0:
            x_ci = x
            f_ci = fstar_ci_param / zm * (1. - zm / pblh) / (umean / ustar * k)
        else:
            return {'x_ci_max': None, 'x_ci': None, 'f_ci': None,
                    'x': None, 'f': None, 'sigy': None, 'f_1d': None,
                    'fstar_ci_param': fstar_ci_param, 'flag_err': 1}

    # Maximum location of influence (peak location)
    xstarmax = -c / b + d
    if z0 is not None:
        x_ci_max = xstarmax * zm / (1. - (zm / pblh)) * (np.log(zm / z0) - psi_f)
    else:
        x_ci_max = xstarmax * zm / (1. - (zm / pblh)) * (umean / ustar * k)

    # Real scale sig_y
    if abs(mo_length) > oln:
        mo_length = -1E6
    if mo_length <= 0:  # convective
        scale_const = 1E-5 * abs(zm / mo_length)**(-1) + 0.80
    elif mo_length > 0:  # stable
        scale_const = 1E-5 * abs(zm / mo_length)**(-1) + 0.55
    if scale_const > 1:
        scale_const = 1.0
    sigy = sigystar_param / scale_const * zm * v_sigma / ustar
    sigy[sigy < 0] = np.nan

    f_1d = np.abs(f_ci * 1 / (np.sqrt(2 * np.pi) * sigy))

    return {'x_ci_max': x_ci_max, 'x_ci': x_ci, 'f_ci': f_ci,
            'x': x, 'f': f_1d, 'flag_err': flag_err,
            'sigy': sigy,
            'f_1d': f_1d, 'fstar_ci_param': fstar_ci_param
            }


#: Provenance stamped into every footprint this model produces.
MODEL_META = {
    "model_citation": ("Kljun, N., P. Calanca, M.W. Rotach, H.P. Schmid "
                       "(2015): The simple two-dimensional parameterisation "
                       "for Flux Footprint Prediction (FFP). Geosci. Model "
                       "Dev. 8, 3695-3713."),
    "model_doi": "10.5194/gmd-8-3695-2015",
    "model_reference_version": "FFP 1.42",
}


#: The registered model is the generic pipeline wrapped around _kljun_record —
#: only the kernel (and the constants inside it) is Kljun physics. The
#: decorator supplies the canonical signature, the driver, provenance, and the
#: kernel-protocol attributes (.kernel/.resolve_grid/.validate/...).
calc = engine.footprint_model(
    "kljun2015", description="Kljun et al. (2015) FFP parameterisation",
    meta=MODEL_META, options=("rslayer",), defaults={"rslayer": 0},
    log=logger)(_kljun_record)

calc.__doc__ = """Kljun et al. (2015) footprint as a :class:`~fluxprint.footprint.Footprint`.

    Accepts scalars (one record) or equal-length sequences (composited into
    one footprint) and returns the result on the model's regular grid in the
    local tower-centred frame. Provide ``wind_dir`` for a north-up
    (geographically oriented) grid.

    Args:
        zm: Measurement height above displacement [m].
        ustar: Friction velocity [m s-1].
        pblh: Boundary-layer height [m].
        mo_length: Obukhov length [m].
        v_sigma: Std. dev. of lateral velocity [m s-1].
        wind_dir: Wind direction [deg]; rotates the footprint to geographic axes.
        z0: Roughness length [m] (or pass ``umean`` instead).
        umean: Mean wind speed [m s-1] (alternative to ``z0``).
        domain: ``[xmin, xmax, ymin, ymax]`` [m]; defaults to a tower-centred 2 km box.
        dx, dy: Grid spacing [m].
        nx, ny: Grid element counts (alternative to ``dx``/``dy``).
        rslayer: Set ``1`` to compute even within the roughness sublayer.
        smooth: Apply the standard smoothing kernel (generic spelling; wins
            over ``smooth_data`` when both are given). Default on.
        smooth_data: FFP-compatible spelling of ``smooth`` (kept for
            downstream integrations).
        tower, tower_crs, time: Metadata attached to the returned footprint.
        verbosity: 2 logs progress, 1 only fatal problems, 0 silent.

    Returns:
        A local-frame :class:`~fluxprint.footprint.Footprint` (``n`` = records
        composited; ``attrs["flag_err"]`` carries the model error flag).
    """
