"""

"""

import numbers
import logging
import numpy as np
from scipy import signal as sg


from ..exceptions import *
from ..footprint import Footprint
from .base import register_model

logger = logging.getLogger('fluxprint.model.kljun_et_al_2015')

# from __future__ import print_function


def calc_ffp_climatology(zm=None, z0=None, umean=None, pblh=None, mo_length=None, v_sigma=None, ustar=None,
                    wind_dir=None, domain=None, dx=None, dy=None, nx=None, ny=None, 
                    rs=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], rslayer=0,
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
        rs           = Percentage of source area for which to provide contours, must be between 10% and 90%. 
                       Can be either a single value (e.g., "80") or a list of values (e.g., "[10, 20, 30]")
                       Expressed either in percentages ("80") or as fractions of 1 ("0.8"). 
                       Default is [10:10:80]. Set to "None" for no output of percentages
        rslayer      = Calculate footprint even if zm within roughness sublayer: set rslayer = 1
                       Note that this only gives a rough estimate of the footprint as the model is not 
                       valid within the roughness sublayer. Default is 0 (i.e. no footprint for within RS).
                       z0 is needed for estimation of the RS.
        smooth_data  = Apply convolution filter to smooth footprint climatology if smooth_data=1 (default)
        crop         = Crop output area to size of the 80% footprint or the largest r given if crop=1
        pulse        = Display progress of footprint calculations every pulse-th footprint (e.g., "100")
        verbosity    = Level of verbosity at run time: 0 = completely silent, 1 = notify only of fatal errors,
                       2 = all notifications

    FFP output
        FFP      = Structure array with footprint climatology data for measurement at [0 0 zm] m
        x_2d	    = x-grid of 2-dimensional footprint [m]
        y_2d	    = y-grid of 2-dimensional footprint [m]
        fclim_2d = Normalised footprint function values of footprint climatology [m-2]
        rs       = Percentage of footprint as in input, if provided
        fr       = Footprint value at r, if r is provided
        xr       = x-array for contour line of r, if r is provided
        yr       = y-array for contour line of r, if r is provided
        n        = Number of footprints calculated and included in footprint climatology
        flag_err = 0 if no error, 1 in case of error, 2 if not all contour plots (rs%) within specified domain,
                   3 if single data points had to be removed (outside validity)

    Created: 19 May 2016 natascha kljun
    Converted from matlab to python, together with Gerardo Fratini, LI-COR Biosciences Inc.
    version: 1.42
    last change: 11/12/2019 Gerardo Fratini, ported to Python 3.x
    Copyright (C) 2015 - 2023 Natascha Kljun
    """

    #===========================================================================
    # Input check
    flag_err = 0
        
    # Check existence of required input pars
    if None in [zm, pblh, mo_length, v_sigma, ustar] or (z0 is None and umean is None):
        # miss_cols = [i for i, c in enumerate([zm, pblh, mo_length, v_sigma, ustar]) if c is None]
        # logger.error(f"Missing columns: {'; '.join(miss_cols) if miss_cols else '`z0` and `umean`.'}")
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

    # Rename lists as now the function expects time series of inputs
    ustars, sigmavs, hs, ols, wind_dirs, zms, z0s, umeans = \
            ustar, v_sigma, pblh, mo_length, wind_dir, zm, z0, umean

    #===========================================================================
    # Define computational domain
    # Check passed values and make some smart assumptions
    if isinstance(dx, numbers.Number) and dy is None: dy = dx
    if isinstance(dy, numbers.Number) and dx is None: dx = dy
    if not all(isinstance(item, numbers.Number) for item in [dx, dy]): dx = dy = None
    if isinstance(nx, int) and ny is None: ny = nx
    if isinstance(ny, int) and nx is None: nx = ny
    if not all(isinstance(item, int) for item in [nx, ny]): nx = ny = None
    if not isinstance(domain, list) or len(domain) != 4: domain = None

    if all(item is None for item in [dx, nx, domain]):
        # If nothing is passed, default domain is a square of 2 Km size centered
        # at the tower with pizel size of 2 meters (hence a 1000x1000 grid)
        domain = [-1000., 1000., -1000., 1000.]
        dx = dy = 2.
        nx = ny = 1000
    elif domain is not None:
        # If domain is passed, it takes the precendence over anything else
        if dx is not None:
            # If dx/dy is passed, takes precendence over nx/ny
            nx = int((domain[1]-domain[0]) / dx)
            ny = int((domain[3]-domain[2]) / dy)
        else:
            # If dx/dy is not passed, use nx/ny (set to 1000 if not passed)
            if nx is None: nx = ny = 1000
            # If dx/dy is not passed, use nx/ny
            dx = (domain[1]-domain[0]) / float(nx)
            dy = (domain[3]-domain[2]) / float(ny)
    elif dx is not None and nx is not None:
        # If domain is not passed but dx/dy and nx/ny are, define domain
        domain = [-nx*dx/2, nx*dx/2, -ny*dy/2, ny*dy/2]
    elif dx is not None:
        # If domain is not passed but dx/dy is, define domain and nx/ny
        domain = [-1000, 1000, -1000, 1000]
        nx = int((domain[1]-domain[0]) / dx)
        ny = int((domain[3]-domain[2]) / dy)
    elif nx is not None:
        # If domain and dx/dy are not passed but nx/ny is, define domain and dx/dy
        domain = [-1000, 1000, -1000, 1000]
        dx = (domain[1]-domain[0]) / float(nx)
        dy = (domain[3]-domain[2]) / float(nx)

    # Put domain into more convenient vars
    xmin, xmax, ymin, ymax = domain

    # Define rslayer if not passed
    if rslayer is None: rslayer = 0

    # Define smooth_data if not passed
    if smooth_data is None: smooth_data = 1

    # Define crop if not passed
    if crop is None: crop = 0

    # Define pulse if not passed
    if pulse == None:
        if ts_len <= 20:
            pulse = 1
        else:
            pulse = int(ts_len / 20)

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

    #===========================================================================
    # Define physical domain in cartesian and polar coordinates
    # Cartesian coordinates
    x = np.linspace(xmin, xmax, nx + 1)
    y = np.linspace(ymin, ymax, ny + 1)
    x_2d, y_2d = np.meshgrid(x, y)

    # Polar coordinates
    # Set theta such that North is pointing upwards and angles increase clockwise
    rho = np.sqrt(x_2d**2 + y_2d**2)
    theta = np.arctan2(x_2d, y_2d)

    # initialize raster for footprint climatology
    fclim_2d = np.zeros(x_2d.shape)

    #===========================================================================
    # Loop on time series

    # Initialize logic array valids to those 'timestamps' for which all inputs are
    # at least present (but not necessarily phisically plausible)
    valids = [True if not any([val is None for val in vals]) else False \
              for vals in zip(ustars, sigmavs, hs, ols, wind_dirs, zms)]

    # if verbosity > 1: logger.info('')
    for ix, (ustar, v_sigma, pblh, mo_length, wind_dir, zm, z0, umean) \
            in enumerate(zip(ustars, sigmavs, hs, ols, wind_dirs, zms, z0s, umeans)):

        # Counter
        if verbosity > 1 and ix % pulse == 0:
            logger.info('Calculating footprint %d of %d', ix + 1, ts_len)

        valids[ix] = check_ffp_inputs(ustar, v_sigma, pblh, mo_length, wind_dir, zm, z0, umean, rslayer, verbosity)

        # If inputs are not valid, skip current footprint
        if not valids[ix]:
            raise_ffp_exception(16, verbosity)
        else:
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
                    flag_err = 3
                    valids[ix] = 0
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

            #===========================================================================
            # Add to footprint climatology raster
            fclim_2d = fclim_2d + f_2d

    #===========================================================================
    # Continue if at least one valid footprint was calculated
    n = sum(valids)
    clevs = None
    if n==0:
        logger.error("No footprint calculated")
        flag_err = 1
    else:
        logger.info(f"{n} footprint calculated")
        #===========================================================================
        # Normalize and smooth footprint climatology
        fclim_2d = fclim_2d / n

        # Truthiness, not `is not None`: smooth_data=0 must actually disable
        # smoothing as documented (it used to smooth anyway).
        if smooth_data:
            # np.matrix is pending removal from numpy; same kernel values.
            skernel  = np.array([[0.05, 0.1, 0.05],
                                 [0.10, 0.4, 0.10],
                                 [0.05, 0.1, 0.05]])
            fclim_2d = sg.convolve2d(fclim_2d,skernel,mode='same')
            fclim_2d = sg.convolve2d(fclim_2d,skernel,mode='same')

        #===========================================================================
        # Crop domain and footprint to the largest rs value
        if crop:
            # Deferred: utils pulls the heavy geo/plotting stack at import.
            from ..utils import get_contour_levels, get_contour_vertices
            rs_dummy = 0.8  # crop to 80%
            clevs = get_contour_levels(fclim_2d, dx, dy, rs_dummy)
            xrs = []
            yrs = []
            xrs, yrs = get_contour_vertices(x_2d, y_2d, fclim_2d, clevs[0][2])

            xrs_crop = [x for x in xrs if x is not None]
            yrs_crop = [x for x in yrs if x is not None]

            dminx = np.floor(min(xrs_crop[-1]))
            dmaxx = np.ceil(max(xrs_crop[-1]))
            dminy = np.floor(min(yrs_crop[-1]))
            dmaxy = np.ceil(max(yrs_crop[-1]))
                
            if dminy>=ymin and dmaxy<=ymax:
                jrange = np.where((y_2d[:,0] >= dminy) & (y_2d[:,0] <= dmaxy))[0]
                jrange = np.concatenate(([jrange[0]-1], jrange, [jrange[-1]+1]))
                jrange = jrange[np.where((jrange>=0) & (jrange<=y_2d.shape[0]))[0]]
            else:
                jrange = np.linspace(0, 1, y_2d.shape[0]-1)
                        
            if dminx>=xmin and dmaxx<=xmax:
                irange = np.where((x_2d[0,:] >= dminx) & (x_2d[0,:] <= dmaxx))[0]
                irange = np.concatenate(([irange[0]-1], irange, [irange[-1]+1]))
                irange = irange[np.where((irange>=0) & (irange<=x_2d.shape[1]))[0]]
            else:
                irange = np.linspace(0, 1, x_2d.shape[1]-1)

            jrange = [[it] for it in jrange]
            x_2d = x_2d[jrange,irange]
            y_2d = y_2d[jrange,irange]
            fclim_2d = fclim_2d[jrange,irange]

            
    #===========================================================================
    # Fill output structure
    return type('var_', (object,), {'x_2d': x_2d, 'y_2d': y_2d, 'fclim_2d': fclim_2d,
                'n': n, 'flag_err': flag_err})


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


@register_model("kljun2015",
                description="Kljun et al. (2015) FFP parameterisation",
                **MODEL_META)
def calc(*, zm, ustar, pblh, mo_length, v_sigma, wind_dir, z0=None, umean=None,
         domain=None, dx=None, dy=None, nx=None, ny=None, rslayer=0,
         smooth_data=1, tower=None, tower_crs=None, time=None, verbosity=0,
         **kwargs) -> Footprint:
    """Kljun et al. (2015) footprint as a :class:`~fluxprint.footprint.Footprint`.

    Thin wrapper over :func:`calc_ffp_climatology`. Accepts scalars (one record)
    or equal-length sequences (composited into one footprint) and returns the
    result on the model's regular grid in the local tower-centred frame. Provide
    ``wind_dir`` for a north-up (geographically oriented) grid.

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
        smooth_data: Apply the standard smoothing kernel (``1``) or not (``0``).
        tower, tower_crs, time: Metadata attached to the returned footprint.
        verbosity: Passed through to the underlying routine.

    Returns:
        A local-frame :class:`~fluxprint.footprint.Footprint` (``n`` = records
        composited; ``attrs["flag_err"]`` carries the model error flag).
    """
    def _listify(value):
        # calc_ffp_climatology wraps non-lists as a single element, so arrays /
        # tuples / Series must be converted to a list; scalars pass through.
        if value is None or isinstance(value, (int, float, list)):
            return value
        return list(value)

    out = calc_ffp_climatology(
        zm=_listify(zm), z0=_listify(z0), umean=_listify(umean),
        ustar=_listify(ustar), pblh=_listify(pblh),
        mo_length=_listify(mo_length), v_sigma=_listify(v_sigma),
        wind_dir=_listify(wind_dir), domain=domain, dx=dx, dy=dy, nx=nx, ny=ny,
        rslayer=rslayer, smooth_data=smooth_data, crop=0, verbosity=verbosity)

    from datetime import datetime as _dt, timezone as _tz
    from ..version import __version__ as _fluxprint_version

    x = np.asarray(out.x_2d)[0, :]
    y = np.asarray(out.y_2d)[:, 0]
    # Provenance: enough metadata to trace a stored footprint back to the
    # code, model and settings that produced it.
    attrs = {
        "model": "kljun2015",
        "flag_err": int(out.flag_err),
        **MODEL_META,
        "fluxprint_version": _fluxprint_version,
        "wind_profile_input": "umean" if z0 is None else "z0",
        "rslayer": int(rslayer),
        "smooth_data": int(bool(smooth_data)),
        "history": (f"{_dt.now(_tz.utc).strftime('%Y-%m-%dT%H:%M:%SZ')} "
                    f"created by fluxprint {_fluxprint_version}, "
                    "model kljun2015"),
    }
    fp = Footprint(
        f=np.asarray(out.fclim_2d), x=x, y=y, time=time,
        tower=tower, tower_crs=tower_crs, n=int(out.n), attrs=attrs)
    if fp.n:
        # The grid integral of the full footprint is 1 by construction, so the
        # captured fraction diagnoses how much flux the domain truncates. It is
        # computed on the returned field, so with smooth_data=1 it also
        # includes the ~1% border mass the smoothing kernel loses.
        captured = fp.total()
        fp.attrs["captured_fraction"] = float(captured)
        if captured < 0.8:
            logger.warning(
                "Footprint domain captures only %.0f%% of the flux; source-"
                "area fractions above that are unreachable. Enlarge `domain` "
                "(or reduce `dx`) to capture more.", captured * 100)
    return fp
