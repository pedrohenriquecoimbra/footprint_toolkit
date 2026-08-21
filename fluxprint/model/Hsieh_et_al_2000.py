import numpy as np
from scipy.stats import norm
from scipy.special import gamma
from scipy.interpolate import griddata
from scipy import signal as sg

def rotate_to_wind(x, y, wind_dir):
    """Rotate coordinates to align with wind direction."""
    theta = np.radians(wind_dir)
    x_rot = x * np.cos(theta) + y * np.sin(theta)
    y_rot = -x * np.sin(theta) + y * np.cos(theta)
    return x_rot, y_rot

def patch_index(patch_map):
    """Get unique indices from patch map."""
    return np.unique(patch_map[~np.isnan(patch_map)])


def patch_ffp(ffp, PatchMap=None, path_max_limit=20):
    # Calculate patch sums if patch map provided
    # Probably not a land cover map if too many classes
    PI = patch_index(PatchMap)
    ffp_patch = {}
    if len(PI) < path_max_limit:
        for i in range(len(PI)):
            if ffp is not None:
                ffp_patch[PI[i]] = np.nansum(
                    ffp * (PatchMap == PI[i]))
    return ffp_patch
                    

def domain_to_2d(domain=None, dx=None, dy=None, nx=None, ny=None):
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
            if nx is None:
                nx = ny = 1000
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

    # ===========================================================================
    # Define physical domain in cartesian and polar coordinates
    # Cartesian coordinates
    x = np.linspace(xmin, xmax, nx + 1)
    y = np.linspace(ymin, ymax, ny + 1)
    x_2d, y_2d = np.meshgrid(x, y)
    return x_2d, y_2d


def calc_ffp_climatology(ustar, mo_length, v_sigma, z0, zm, wind_dir, smooth_data=0, **kwargs):
    clim_ffp = type('var_', (object,), {'x_2d': [], 'y_2d': [], 'fclim_2d': 0,
                                        'dist_max': [],
                                        'n': 0, 'flag_err': []})
    for us_, l_, vs_, z0_, zm_, wd_ in list(zip(ustar, mo_length, v_sigma, z0, zm, wind_dir)):
        this_ffp = calc_ffp(us_, l_, vs_, z0_, zm_, wd_, **kwargs)
        # NOT RIGHT, CONFIRM X_2D AND Y_2D AND FCLIM_2D ROTATION 
        clim_ffp.x_2d = this_ffp.x_2d
        clim_ffp.y_2d = this_ffp.y_2d
        clim_ffp.fclim_2d += this_ffp.fclim_2d
        clim_ffp.dist_max += [this_ffp.dist_max]
        clim_ffp.n += 1
        clim_ffp.flag_err += [this_ffp.flag_err]
    
    if clim_ffp.n > 0:
        # Normalize and smooth footprint climatology
        clim_ffp.fclim_2d = clim_ffp.fclim_2d / clim_ffp.n

        # Truthiness, not `is not None`: smooth_data=0 must disable smoothing.
        if smooth_data:
            skernel = np.matrix('0.05 0.1 0.05; 0.1 0.4 0.1; 0.05 0.1 0.05')
            clim_ffp.fclim_2d = sg.convolve2d(
                clim_ffp.fclim_2d, skernel, mode='same')
            clim_ffp.fclim_2d = sg.convolve2d(
                clim_ffp.fclim_2d, skernel, mode='same')
    return clim_ffp
    
def calc_ffp(ustar, mo_length, v_sigma, z0, zm, wind_dir, domain=None, dx=None, dy=None, nx=None, ny=None,
             mscale=0, contour_marks=None, Tdist=0, 
             Twindir=0, ubar=0, pblh=0, **kwargs):
    """
    Calculate footprint using Hsieh et al. (2000) model.
    
    Parameters:
        ustar: friction velocity [m/s]
        mo_length: Obukhov stability length [m]
        v_sigma: fluctuation of lateral wind [m/s] (approx. 2 times ustar)
        z0: momentum roughness length [m]
        zm: height of the measurement [m]
        d: displacement height [m]
        wind_dir: wind direction, from north, degrees (wind coming FROM)
        FX, FY: meshgrid coordinates of mask map
        mscale: 0 for linear scaling, 1 for exponential
    """
    ustar = np.array(ustar)
    mo_length = np.array(mo_length)
    v_sigma = np.array(v_sigma)
    wind_dir = np.array(wind_dir)
    z0 = np.array(z0)
    zm = np.array(zm)

    # Overall Parameters
    # zm = zH - d  # Effective height
    Lx = 100 * zm  # Initial length of along-wind footprint
    k = 0.4  # von Karman constant
    zL = zm / mo_length  # Stability coefficient

    # # Rotate target point
    # Txmax, Tymax = rotate_to_wind(Tdist, 0, Twindir)
    
    # Prepare coordinate systems
    x_2d, y_2d = domain_to_2d(domain, dx, dy, nx, ny)

    Fxprime, Fyprime = rotate_to_wind(x_2d, y_2d, wind_dir)
    # Rl = len(FY)

    # Lateral dispersion width
    ywidth0 = np.floor(z0 * (0.3 * (v_sigma/ustar) * (Lx/z0)**0.86) / 1.5)
    ywidth = ywidth0 * 3
    
    # Hsieh model parameters
    zu = zm * (np.log(zm/z0) - 1 + z0/zm)  # New height scale, Eq 13.5 Hsieh (2000)
    
    # Coefficients D and P according to stability
    P = [0.59, 1, 1.33]
    D = [0.28, 0.97, 2.44]
    
    stab = zu / mo_length
    thresh = 0.04
    
    if stab < -thresh:
        ii = 0
    elif abs(stab) < thresh:
        ii = 1
    elif stab > thresh:
        ii = 2
    
    D1 = D[ii]
    P1 = P[ii]
    
    F2H = (D1 / (0.105 * k**2)) * (zm**-1 * abs(mo_length)**(1-P1) * zu**P1)  # Eq 20, Hsieh
    Xm = np.ceil(F2H * zm)

    # # Initialize all models to None
    # footHsieh = None

    # # Calculate pixel values at target point
    # pxR = {}
    # closestIndexX = np.argmin(np.abs(FX2[0, :] - Txmax))
    # closestIndexY = np.argmin(np.abs(FY2[::-1, 0] - Tymax))

    # if footHsieh is not None:
    #     pxR['H'] = footHsieh[closestIndexY, closestIndexX]

    flag_err = 0

    if mscale == 0:  # linear
        bin_size = max(1, np.round(F2H * zm / 500, 0))
        x = np.arange(0.001, Xm + bin_size, bin_size)  # Avoid eps
    elif mscale == 1:  # exponential
        bin_size = 0.1
        x = np.exp(np.arange(0, np.log(Xm * 1.1) + bin_size))

    # Cross-wind integrated footprint (Eq 17 Hsieh)
    T2 = (-1 / k**2) * (D1 * zu**P1 * abs(mo_length)**(1-P1)) / x
    Fp = -(T2 / x) * np.exp(T2)
    
    # 2D expansion (Detto 2006)
    nn = len(x)
    sy = z0 * 0.3 * (v_sigma/ustar) * (x/z0)**0.86  # Eq B4 Detto
    
    if mscale == 0:  # linear
        y = np.arange(-ywidth, ywidth + bin_size, bin_size)
    elif mscale == 1:  # exponential
        y_pos = np.exp(np.arange(0, np.log(ywidth) + bin_size))
        y = np.concatenate([-np.flip(y_pos), [0], y_pos])
    
    # Create footprint matrix
    footo = np.zeros((nn, len(y)))
    for i in range(nn):
        footo[i,:] = Fp[i] * norm.pdf(y, 0, sy[i])
    
    # Unrotated footprint
    foot = footo.T

    xx, yy = np.meshgrid(x, y)

    if not np.all(y == 0):
        foot_rotate = griddata((xx.flatten(), yy.flatten()), foot.flatten(),
                                (Fxprime.flatten(), Fyprime.flatten()), 
                                method='linear')
        foot_rotate = foot_rotate.reshape(x_2d.shape)
        
        if mscale == 0:
            dx = x[1] - x[0]
            dy = y[1] - y[0]
            binnorm = (np.nansum(foot) * dx * dy) / np.nansum(foot_rotate)
        elif mscale == 1:
            dx = np.concatenate([[x[0]], np.diff(x)])
            dy = np.concatenate([[abs(y[1]-y[0])], np.diff(y)])
            dxx, dyy = np.meshgrid(dx, dy)
            DD = foot * dxx * dyy
            binnorm = np.nansum(DD) / np.nansum(foot_rotate)
        
        ffp_2d = foot_rotate * binnorm
        
        # Distance of maximum contribution (Eq 19 Hsieh)
        HXmax = D1 * zu**P1 * abs(mo_length)**(1-P1) / (2 * 0.4**2)
        Hxmax, Hymax = rotate_to_wind(HXmax, 0, wind_dir)
    else:
        ffp_2d = np.zeros((1,1)) * np.nan
        HXmax = np.nan
        flag_err = 1
    
    return type('var_', (object,), {'x_2d': Fxprime, 'y_2d': Fyprime, 'fclim_2d': ffp_2d.T,
                                    'dist_max': HXmax,
                                    'n': 1, 'flag_err': flag_err})
