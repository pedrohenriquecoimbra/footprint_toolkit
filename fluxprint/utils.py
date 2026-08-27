# built-in modules
import warnings
import logging
import copy

# 3rd party modules
import numpy as np
from numpy import ma
import xarray as xr
from pyproj import Transformer  # for coordinate transformations
import rasterio
import matplotlib.pyplot as plt
from scipy import signal as sg

# local modules
from .commons import update_nested_dict
from .template import DEFAULT_ATTRS

logger = logging.getLogger('fluxprint.utils')

class structuredData:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            self.__dict__[k] = v
        pass


def convert_to_object(data, name=None):
    # Convert data to dictionary
    # footprint = {}
    footprint = type('var_', (object,), {
        'x_2d': None, 'y_2d': None, 'fclim_2d': None,
        'dist_max': None, 'n': None, 'flag_err': None})
    if isinstance(data, (xr.Dataset, xr.DataArray)):
        # Convert xarray Dataset to dictionary
        # if len(data.footprint.dims) == 2:
        #     data = {name: data}

        footprint.group = data[data.footprint.dims[0]]
        footprint.fclim_2d = data.footprint.values
        x_2d, y_2d = np.meshgrid(
            data.x.values, data.y.values)
        footprint.x_2d = x_2d
        footprint.y_2d = y_2d

        # for i, name in enumerate(data[data.footprint.dims[0]].values):
        #     footprint = {
        #         'group': footprint.get('group', []) + [name],
        #         'fclim_2d': footprint.get('group', []) + [data['footprint'].values[i]]
        #     }

        # footprint[name]["x_2d"], footprint[name]["y_2d"] = np.meshgrid(
        #     data['x'].values, data['y'].values)
        return footprint
    elif isinstance(data, (rasterio.io.DatasetWriter, rasterio.io.DatasetReader)):
        if name is None:
            name = data.name

        # Read the first band of the raster data
        footprint_data = data.read(1)  # Use `read(1)` to read the first band

        # Generate x and y coordinates based on the raster bounds and shape
        x = np.linspace(data.bounds.left, data.bounds.right,
                        footprint_data.shape[1])
        y = np.linspace(data.bounds.bottom, data.bounds.top,
                        footprint_data.shape[0])
        x, y = np.meshgrid(x, y)

        # Store the footprint data and coordinates in a dictionary
        footprint = {
            'group': name,
            'fclim_2d': footprint_data,
            'x_2d': x,
            'y_2d': y
        }
        return type('var_', (object,), footprint)

        # Convert rasterio DatasetReader to dictionary
        footprint = {'fclim_2d': data.read()[0]}
        x = np.linspace(data.bounds[0], data.bounds[2],
                        footprint['fclim_2d'].shape[0])
        y = np.linspace(data.bounds[1], data.bounds[3],
                        footprint['fclim_2d'].shape[1])
        footprint["x_2d"], footprint["y_2d"] = np.meshgrid(x, y)
        return {name: footprint}
    else:
        raise ValueError(
            "Data must be a netcdf (xr array) or a rasterio Dataset.")


def convert_to_nc(data, **attrs):
    if isinstance(data, xr.Dataset):
        logger.debug('Data is already in netcdf format.')
        attrs = update_nested_dict(data.attrs, attrs)
    elif isinstance(data, (rasterio.io.DatasetWriter, rasterio.io.DatasetReader)):
        # Convert rasterio DatasetReader to xarray Dataset
        data = convert_to_object(data)
        # Convert dictionary to xarray Dataset
        data = object_to_nc(data)
    elif hasattr(data, "fclim_2d"):
        # Footprint object (carries an `fclim_2d` attribute) -> xarray Dataset
        data = object_to_nc(data)
    else:
        raise TypeError(
            "Data must be an xarray Dataset, a rasterio dataset, or a footprint "
            f"object with an `fclim_2d` attribute; got {type(data).__name__}.")

    # Update attributes
    attrs = update_nested_dict(
        copy.deepcopy(DEFAULT_ATTRS), attrs)
    data = update_attrs_in_nc(data, attrs)
    return data


def convert_to_tif(data, anchor='top-left', **attrs):
    """
    Save footprint data as a TIFF file.
    
    Parameters:
        footprint (object): Footprint data.
        output_path (str): Path to save the TIFF file.
        crs (str): Coordinate reference system.
    """
    if isinstance(data, xr.Dataset):
        # Convert xarray Dataset to rasterio Dataset
        arr = data['footprint'].to_numpy()
        x = data['x'].to_numpy()
        y = data['y'].to_numpy()
        dx = data['x'].attrs.get('dx', data.attrs.get('dx', 10))
        dy = data['y'].attrs.get('dy', data.attrs.get('dy', dx))
        crs = extract_crs(data)

        # Footprint into array (band, lon, lat)
        if len(arr.shape) < 3:
            arr = np.array([arr])

        convention, anchor = infer_convention_from_nc(data)
        transform = affine_conventions(
            x, y, dx, dy, anchor=anchor, convention=convention)

        # Write a .tif profile
        profile = {'driver': 'GTiff',
                   'dtype': arr.dtype,
                   'nodata': None,
                   'width': arr.shape[2],
                   'height': arr.shape[1],
                   'count': arr.shape[0],
                   'crs': crs,
                   'transform': transform,
                   'compress': 'lzw',
                   }
        profile.update(
            {k: v for k, v in data.attrs.items() if k in profile.keys()})
        profile.update({k: v for k, v in attrs.items() if k in profile.keys()})

        memory_tif = rasterio.io.MemoryFile().open(**profile)
        [memory_tif.write(band, b + 1) for b, band in enumerate(arr)]

        return memory_tif

    elif hasattr(data, "fclim_2d"):
        # Footprint object -> Dataset (with attrs/CRS merged) -> recurse to the
        # xarray branch above, which actually builds and returns the TIFF.
        # Previously this branch fell through to `return None`.
        return convert_to_tif(convert_to_nc(data, **attrs), anchor=anchor)
    else:
        raise TypeError(
            "Data must be an xarray Dataset, a rasterio dataset, or a footprint "
            f"object with an `fclim_2d` attribute; got {type(data).__name__}.")
    return


# utils.py
def is_footprint_dict(d):
    return ('fclim_2d' in d) or (len(d.keys()) and 'fclim_2d' in d[list(d.keys())[0]])


def find_utm_epsg_from_lon_deprecated(lon: float, lat: float = None):
    utm_band = str(int(np.floor((lon + 180) / 6) % 60))
    epsg_code = "EPSG:327" + utm_band.zfill(2)
    return epsg_code

def find_peak(array):
    return np.unravel_index(array.argmax(), array.shape)


def smooth_data(data):
    """Deprecated alias of :func:`fluxprint.footprint.smooth_field`."""
    warnings.warn(
        "fluxprint.utils.smooth_data is deprecated and will be removed in a "
        "future release; use fluxprint.footprint.smooth_field() instead.",
        DeprecationWarning, stacklevel=2)
    from .footprint import smooth_field
    return smooth_field(data)

def transform_crs(*xy, crs_in="EPSG:4326", crs_out="EPSG:3035"):
    transformer = Transformer.from_crs(crs_in, crs_out)
    return transformer.transform(*xy)

def find_utm_epsg_from_lon(lon: float, lat: float = None) -> str:
    """
    Determine the UTM EPSG code based on longitude.

    Parameters:
        lon (float): Longitude in decimal degrees.
        lat (float, optional): Latitude in decimal degrees. Not used in the calculation.

    Returns:
        str: UTM EPSG code (e.g., "EPSG:32612" for UTM zone 12N).
    """
    # Calculate UTM zone number
    utm_zone = int(np.floor((lon + 180) / 6) + 1)

    # Determine hemisphere (N or S)
    hemisphere="6" if lat is None or lat >= 0 else "7"

    # Construct EPSG code
    epsg_code=f"EPSG:32{hemisphere}{utm_zone:02d}"
    return epsg_code


def update_affine(src, a=0, b=0, c=0, d=0, e=0, f=0):
    # Get the affine transformation matrix
    transform = src.transform

    # Modify the affine transformation matrix to shift the raster north by some distance
    # The affine matrix is in the form:
    # | a, b, c |
    # | d, e, f |
    # Where (c, f) are the top-left corner coordinates (longitude, latitude)
    # To shift north by 1°, add 1 to the latitude (f)
    new_transform = rasterio.transform.guard_transform(
        (transform[0] + a, transform[1] + b, transform[2] + c,
         transform[3] + d, transform[4] + e, transform[5] + f))

    # Get the metadata of the input raster
    # Update the metadata with the new affine transformation matrix
    src.transform = new_transform
    return src


def update_attrs_in_nc(data, attrs={}):
    """Update attributes in a NetCDF dataset.
    
    Args:
        data (xr.Dataset): The dataset to update.
        attrs (object): The attributes to update.
        
    Returns:
        xr.Dataset: The updated dataset.
    """
    # Update global attributes
    global_attrs = copy.deepcopy(attrs.pop('__global__', {}))
    for k, v in attrs.items():
        if k not in data.variables.keys():
            global_attrs.update({k: v})
    data.attrs.update(global_attrs)

    # Update variable attributes
    for k, v in attrs.items():
        if k in data.variables.keys():
            data[k].attrs.update(v)
    return data


def fp_to_nc(fclim_2d, x, y, timestep, attrs={}):
    data = xr.Dataset({'footprint': (('timestep', 'x', 'y'), np.array(fclim_2d))},
                      coords={'timestep': np.array(timestep), 'x': np.array(x), 'y': np.array(y)})
    data = update_attrs_in_nc(data, attrs)
    return data


def object_to_nc(footprint, attrs={}):
    if footprint == {}:
        return None
    x = np.array(footprint.x_2d)[0, 0, :]
    y = np.array(footprint.y_2d)[0, :, 0]
    footprints = footprint.fclim_2d
    timesteps = footprint.group
    data = fp_to_nc(footprints, x, y, timesteps, attrs=attrs)
    return data

def transformer_convention(data, convention, anchor, convention_from: str = None, anchor_from: str = None):
    if isinstance(data, xr.Dataset):
        convention_infer, anchor_infer = infer_convention_from_nc(data)
    elif isinstance(data, (rasterio.io.DatasetWriter, rasterio.io.DatasetReader)):
        convention_infer, anchor_infer = infer_convention_from_affine(
            data.transform)
    anchor_from = anchor_from or anchor_infer
    convention_from = convention_from or convention_infer

    # Remove spaces
    anchor_from = anchor_from.replace(' ', '')
    convention_from = convention_from.replace(' ', '')
    anchor = anchor.replace(' ', '')
    convention = convention.replace(' ', '')

    if anchor_from != anchor_infer.replace(' ', ''):
        warnings.warn(
            f'Anchor `{anchor_from}` differs from infered anchor `{anchor_infer}`.')
    if convention_from != convention_infer.replace(' ', ''):
        warnings.warn(
            f'Convention `{convention_from}` differs from infered convention `{convention_infer}`.')

    func_ = []
    if convention != convention_from:
        logger.debug('def f_(arr): return arr.T')
        def f_(arr): return arr.T
        func_.append(f_)

    if anchor.split('-')[1] != anchor_from.split('-')[1]:
        logger.debug('def f_(arr): return arr[:, :, ::-1]')
        def f_(arr): return arr[:, :, ::-1] if len(arr.shape) == 3 else arr[:, ::-1]
        func_.append(f_)
    if anchor.split('-')[0] != anchor_from.split('-')[0]:
        logger.debug('def f_(arr): return arr[:, ::-1, :]')
        def f_(arr): return arr[:, ::-1, :] if len(arr.shape) == 3 else arr[::-1, :]
        func_.append(f_)

    # if convention == '(lon,lat)' and anchor.split('-')[0] != anchor_from.split('-')[0]:
    #     logger.debug('def f_(arr): return arr[:, :, ::-1]')
    #     def f_(arr): return arr[:, :, ::-1] if len(arr.shape) == 3 else arr[:, ::-1]
    #     func_.append(f_)
    
    # if convention == '(lon,lat)' and anchor.split('-')[1] != anchor_from.split('-')[1]:
    #     logger.debug('def f_(arr): return arr[:, ::-1, :]')
    #     def f_(arr): return arr[:, ::-1, :] if len(arr.shape) == 3 else arr[::-1, :]
    #     func_.append(f_)
    
    # if convention == '(lat,lon)' and anchor.split('-')[0] != anchor_from.split('-')[0]:
    #     logger.debug('def f_(arr): return arr[:, ::-1, :]')
    #     def f_(arr): return arr[:, ::-1, :] if len(arr.shape) == 3 else arr[::-1, :]
    #     func_.append(f_)
    
    # if convention == '(lat,lon)' and anchor.split('-')[1] != anchor_from.split('-')[1]:
    #     logger.debug('def f_(arr): return arr[:, :, ::-1]')
    #     def f_(arr): return arr[:, :, ::-1] if len(arr.shape) == 3 else arr[:, ::-1]
    #     func_.append(f_)


    def supra_function(array):
        for func in func_:
            array = func(array)
        return array
    return supra_function


def find_middle_point(bounds):
    # Get the bounding box
    min_x, min_y, max_x, max_y = bounds.left, bounds.bottom, bounds.right, bounds.top

    # Calculate the middle point
    middle_x = (min_x + max_x) / 2
    middle_y = (min_y + max_y) / 2

    return middle_y, middle_x
    
def affine_conventions(x, y, dx, dy, anchor='top-left', convention='(lat,lon)'):
    default = rasterio.Affine(0, dx, np.min(x),   dy, 0, np.min(y))
    transform_affine = {
        '(lat,lon)': {'default':      rasterio.Affine(dx, 0, np.min(x),   0, -dy, np.max(y)),
                  'top-left':     rasterio.Affine(dx, 0, np.min(x),   0, -dy, np.max(y)),
                  'top-right':    rasterio.Affine(-dx, 0, np.max(x),  0, -dy, np.max(y)),
                  'bottom-right': rasterio.Affine(-dx, 0, np.max(x),  0, dy, np.min(y)),
                  'bottom-left':  rasterio.Affine(dx, 0, np.min(x),   0, dy, np.min(y)), },
        '(lon,lat)': {'default':      rasterio.Affine(0, dx, np.min(x),   dy, 0, np.min(y)),
                      'top-left':     rasterio.Affine(0, dx, np.min(x),   -dy, 0, np.max(y)),
                      'top-right':    rasterio.Affine(0, -dx, np.max(x),  -dy, 0, np.max(y)),
                      'bottom-right': rasterio.Affine(0, -dx, np.max(x),  dy, 0, np.min(y)),
                      'bottom-left':  rasterio.Affine(0, dx, np.min(x),   dy, 0, np.min(y)), }, }
    
    if convention in transform_affine:
        if not transform_affine.get(convention).get(anchor, None):
            warnings.warn(
                f'Ignoring anchor {anchor} and convention ({convention}) and using default affine.')
        return transform_affine.get(convention).get(anchor, transform_affine[convention]['default'])
    else:
        if anchor or convention:
            warnings.warn(
                f'Ignoring anchor {anchor} and convention ({convention}) and using default affine.')
        return default


def identify_convention(affine_matrix, dx, dy, min_x, max_x, min_y, max_y):
    a, b, xoff, d, e, yoff = affine_matrix[:6]

    if a == dx and e == -dy and xoff == min_x and yoff == max_y:
        return "(lat,lon)", "top-left"
    elif a == -dx and e == -dy and xoff == max_x and yoff == max_y:
        return "(lat,lon)", "top-right"
    elif a == -dx and e == dy and xoff == max_x and yoff == min_y:
        return "(lat,lon)", "bottom-right"
    elif a == dx and e == dy and xoff == min_x and yoff == min_y:
        return "(lat,lon)", "bottom-left"
    
    elif b == dx and d == dy and xoff == min_x and yoff == min_y:
        return "(lon,lat)", "bottom-left"
    elif b == -dx and d == dy and xoff == min_x and yoff == min_y:
        return "(lon,lat)", "bottom-right"
    elif b == -dx and d == -dy and xoff == min_x and yoff == min_y:
        return "(lon,lat)", "top-right"
    elif b == dx and d == -dy and xoff == min_x and yoff == min_y:
        return "(lon,lat)", "top-left"
    else:
        return "Unknown convention"


def infer_convention_from_affine(affine_matrix):
    order = "Unknown convention"
    anchor = "Unknown convention"

    a, b, xoff, d, e, yoff = affine_matrix[:6]

    if a > 0 and e < 0:
        return "(lat,lon)", "top-left"
    elif a < 0 and e < 0:
        return "(lat,lon)", "top-right"
    elif a < 0 and e > 0:
        return "(lat,lon)", "bottom-right"
    elif a > 0 and e > 0:
        return "(lat,lon)", "bottom-left"
    elif b > 0 and d > 0:
        return "(lon,lat)", "bottom-left"
    elif b > 0 and d < 0:
        return "(lon,lat)", "bottom-right"
    elif b < 0 and d < 0:
        return "(lon,lat)", "top-right"
    elif b < 0 and d > 0:
        return "(lon,lat)", "top-left"
    else:
        return order, anchor


def infer_convention_from_nc(data):
    order = "Unknown convention"
    anchor = "Unknown convention"

    if 'x' in list(data.footprint.dims)[-2] and 'y' in list(data.footprint.dims)[-1]:
        a = data.x[-1] - data.x[0]
        e = data.y[-1] - data.y[0]
        b = d = 0
        order = "(lon, lat)"
    else:
        a = e = 0
        b = data.x[-1] - data.x[0]
        d = data.y[-1] - data.y[0]
        order = "(lat,lon)"
    if e < 0 or b < 0:
        anchor = "top-"
    elif e > 0 or b > 0:
        anchor = "bottom-"
    if a > 0 or d > 0:
        anchor += "left"
    elif a < 0 or d < 0:
        anchor += "right"
    # if a > 0 and e < 0:
    #     return order, "top-left"
    # elif a < 0 and e < 0:
    #     return order, "top-right"
    # elif a < 0 and e > 0:
    #     return order, "bottom-right"
    # elif a > 0 and e > 0:
    #     return order, "bottom-left"
    # elif b > 0 and d > 0:
    #     return order, "bottom-left"
    # elif b > 0 and d < 0:
    #     return order, "bottom-right"
    # elif b < 0 and d < 0:
    #     return order, "top-right"
    # elif b < 0 and d > 0:
    #     return order, "top-left"
    # else:
    return order, anchor

def transform_coordinates(*args, crs_in="EPSG:4326", crs_out="EPSG:3035"):
    """
    Transform coordinates between CRS.
    
    Parameters:
        args (float): Coordinates.
        crs_in (str): Input CRS.
        crs_out (str): Output CRS.
    
    Returns:
        tuple: Transformed coordinates.
    """
    transformer = Transformer.from_crs(crs_in, crs_out)
    return transformer.transform(*args)


def attribute_crs(data, crs="EPSG:3035"):
    if isinstance(data, (xr.Dataset, xr.DataArray)):
        crs = rasterio.crs.CRS.from_string(crs)
        new_attrs = {'__global__': {'Coordinate_Reference_System': crs.to_string(),
                            'crs_projection4': crs.to_proj4(),
                            'crs_wkt': crs.to_wkt()}
                    }
        data = update_attrs_in_nc(data, new_attrs)
    return data

def extract_crs(nc, str:bool=False):
    if nc.attrs.get('crs_wkt', None):
        crs = rasterio.crs.CRS.from_wkt(nc.attrs['crs_wkt'])
        # if str:
        #     crs = f'EPSG:{crs.to_epsg() or '4326'}'
    else:
        crs = nc.attrs.get('crs', None)
    return crs


def reproject_tif(src, crs):
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    transform, width, height = calculate_default_transform(
        src.crs, crs, src.width, src.height, *src.bounds)
    kwargs = src.meta.copy()

    kwargs.update({
        'crs': crs,
        'transform': transform,
        'width': width,
        'height': height})

    dst = rasterio.io.MemoryFile().open(**kwargs)
    for i in range(1, src.count + 1):
        reproject(
            source=rasterio.band(src, i),
            destination=rasterio.band(dst, i),
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=crs,
            resampling=Resampling.nearest)
    return dst


def reproject_netcdf(data, crs_to, crs_from=None):
    crs_from = extract_crs(data, str=True) or crs_from
    logger.debug(f'Reproject_netcdf from {crs_from} to {crs_to}.')
    coords = list(zip(*[transform_coordinates(
        *coord, crs_in=crs_from, crs_out=crs_to) for coord in list(zip(data.y.to_numpy(), data.x.to_numpy()))]))
    data['x'] = (data['x'].dims, np.array(coords[1]), data['x'].attrs)
    data['y'] = (data['y'].dims, np.array(coords[0]), data['y'].attrs)
    crs = rasterio.crs.CRS.from_string(crs_to)
    data.attrs.update({'Coordinate_Reference_System': crs.to_string(),
                       'crs_projection4': crs.to_proj4(),
                       'crs_wkt': crs.to_wkt()})
    return data


def get_contour_levels(f, dx, dy, rs=None):
    '''Contour levels of f at percentages of f-integral given by rs
    For original see Kljun, N., P. Calanca, M.W. Rotach, H.P. Schmid, 2015 (doi:10.5194/gmd-8-3695-2015)
    '''
    # Check input and resolve to default levels in needed
    if not isinstance(rs, (int, float, list)):
        rs = list(np.linspace(0.10, 0.90, 9))
    if isinstance(rs, (int, float)):
        rs = [rs]

    # Levels
    pclevs = np.empty(len(rs))
    pclevs[:] = np.nan
    ars = np.empty(len(rs))
    ars[:] = np.nan

    sf = np.sort(f, axis=None)[::-1]
    # Masked array for handling potential nan
    msf = ma.masked_array(sf, mask=(np.isnan(sf) | np.isinf(sf)))

    csf = msf.cumsum().filled(np.nan)*dx*dy
    for ix, r in enumerate(rs):
        dcsf = np.abs(csf - r)
        pclevs[ix] = sf[np.nanargmin(dcsf)]
        ars[ix] = csf[np.nanargmin(dcsf)]

    return [(round(r, 3), ar, pclev) for r, ar, pclev in zip(rs, ars, pclevs)]


def get_contour_vertices(x, y, f, lev):
    '''Contour vertices of f at percentages of f-integral given by rs
    For original see Kljun, N., P. Calanca, M.W. Rotach, H.P. Schmid, 2015 (doi:10.5194/gmd-8-3695-2015)
    '''
    cs = plt.contour(x, y, f, [lev])
    plt.close()
    segs = cs.allsegs[0][0]
    xr = [vert[0] for vert in segs]
    yr = [vert[1] for vert in segs]
    # Set contour to None if it's found to reach the physical domain
    if x.min() >= min(segs[:, 0]) or max(segs[:, 0]) >= x.max() or \
       y.min() >= min(segs[:, 1]) or max(segs[:, 1]) >= y.max():
        return [None, None]

    return [xr, yr]   # x,y coords of contour points.


def center_footprint(footprint, centre=None, center_previous=None):
    def cy(x): return None if x is None else list(map(cy, x)) if (
        isinstance(x, list) or len(x.shape) > 1) else x + centre[0] - center_previous[0]

    def cx(x): return None if x is None else list(map(cx, x)) if (
        isinstance(x, list) or len(x.shape) > 1) else x + centre[1] - center_previous[1]
    
    # def center_object(footprint):
    #     footprint = copy.deepcopy(footprint)
    #     return footprint
    
    if isinstance(footprint, xr.Dataset):
        if not centre:
            centre = (
                float(footprint.attrs.get('Tower_Location_Latitude', 0)), 
                float(footprint.attrs.get('Tower_Location_Longitude', 0)))
            centre_crs = footprint.attrs.get('Tower_Location_CRS', "EPSG:4326")
            crs = extract_crs(footprint) or "EPSG:3035"
            centre = transform_coordinates(*centre, crs_in=centre_crs, crs_out=crs)
        if not center_previous:
            center_previous = (
                np.nanmean([footprint.y.max(), footprint.y.min()]),
                np.nanmean([footprint.x.max(), footprint.x.min()])
            )
        
        # guarantee/warn everything in meters

        # f_ = convert_to_object(footprint)
        # f_ = {k: __center_footprint__(v, centre, center_previous) for k, v in f_.items()}
        # f_ = convert_to_nc(f_)

        x_attrs = footprint.coords['x'].attrs.copy()
        y_attrs = footprint.coords['y'].attrs.copy()
        footprint.coords['x'] = cx(footprint.x)
        footprint.coords['y'] = cy(footprint.y)
        footprint.coords['x'].attrs = x_attrs
        footprint.coords['y'].attrs = y_attrs
        # footprint = footprint.assign_coords(x=cx(footprint.x),
        #                         y=cy(footprint.y))

    elif isinstance(footprint, (rasterio.io.DatasetWriter, rasterio.io.DatasetReader)):
        assert centre and center_previous, 'Centre not defined.'
        footprint = update_affine(
            footprint, c=centre[1] - center_previous[1], 
            f=centre[0] - center_previous[0])

    elif isinstance(footprint, object):
        # f = convert_to_object(footprint)
        # centre
        center_previous = (0, 0)
        
        for var in set(['xr', 'x_2d']) & set(vars(footprint).keys()):
            footprint[var] = list(map(cx, footprint[var]))

        for var in set(['yr', 'y_2d']) & set(vars(footprint).keys()):
            footprint[var] = list(map(cy, vars(footprint)[var]))

        for var in ['x_2d', 'y_2d']:
            footprint[var] = np.array(footprint[var])
    return footprint

def plot_footprint(x_2d, y_2d, fs, clevs=None, show_heatmap=True, normalize=None, 
                   colormap=None, line_width=0.5, iso_labels=None):
    '''Plot footprint function and contours if request'''

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.colors import LogNorm

    # If input is a list of footprints, don't show footprint but only contours,
    # with different colors
    if isinstance(fs, list):
        show_heatmap = False
    else:
        fs = [fs]

    if colormap is None: colormap = cm.jet
    # Define colors for each contour set
    cs = [colormap(ix) for ix in np.linspace(0, 1, len(fs))]

    # Initialize figure
    fig, ax = plt.subplots(figsize=(10, 8))
    # fig.patch.set_facecolor('none')
    # ax.patch.set_facecolor('none')

    if clevs is not None:
        # Temporary patch for pyplot.contour requiring contours to be in ascending orders
        clevs = clevs[::-1]

        # Eliminate contour levels that were set to None
        # (e.g. because they extend beyond the defined domain)
        clevs = [clev for clev in clevs if clev is not None]

        # Plot contour levels of all passed footprints
        # Plot isopleth
        levs = [clev for clev in clevs]
        for f, c in zip(fs, cs):
            cc = [c]*len(levs)
            if show_heatmap:
                cp = ax.contour(x_2d, y_2d, f, levs, colors = 'w', linewidths=line_width)
            else:
                cp = ax.contour(x_2d, y_2d, f, levs, colors = cc, linewidths=line_width)
            # Isopleth Labels
            if iso_labels is not None:
                pers = [str(int(clev[0]*100))+'%' for clev in clevs]
                fmt = {}
                for l,s in zip(cp.levels, pers):
                    fmt[l] = s
                plt.clabel(cp, cp.levels[:], inline=1, fmt=fmt, fontsize=7)

    # plot footprint heatmap if requested and if only one footprint is passed
    if show_heatmap:
        if normalize == 'log':
            norm = LogNorm()
        else:
            norm = None

        xmin = np.nanmin(x_2d)
        xmax = np.nanmax(x_2d)
        ymin = np.nanmin(y_2d)
        ymax = np.nanmax(y_2d)
        for f in fs:
            im = ax.imshow(f[:, :], cmap=colormap, extent=(xmin, xmax, ymin, ymax),
                           norm=norm, origin='lower', aspect=1)
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')

        # for f in fs:
        #     pcol = plt.pcolormesh(x_2d, y_2d, f, cmap=colormap, norm=norm)
        # plt.xlabel('x [m]')
        # plt.ylabel('y [m]')
        # plt.gca().set_aspect('equal', 'box')

        # Colorbar
        cbar = fig.colorbar(im, shrink=1.0, format='%.3e')
        #cbar.set_label('Flux contribution', color = 'k')
    plt.show()

    return fig, ax


def get_colormap(index, total):
    # Define base RGB colors to cycle through or interpolate
    base_colors = [
        (1, 0, 0),  # Red
        (0, 1, 0),  # Green
        (0, 0, 1),  # Blue
        (1, 1, 0),  # Yellow
        (1, 0, 1),  # Magenta
        (0, 1, 1),  # Cyan
    ]

    # Cycle through or interpolate colors if more figures than base_colors
    if total <= len(base_colors):
        base_color = base_colors[index]
    else:
        # Interpolate between colors if more than base
        r = index / max(1, total - 1)
        base_color = plt.cm.hsv(r)[:3]  # Use HSV colormap for variety

    # Create a colormap varying alpha from 0 to 1
    n = 256
    colors = (base_color[0], base_color[1], base_color[2])
    return colors

def plot_leaflet(*ncs, dst=None, labels=[], editable=False):
    """
    Input
        dst='sample/output/footprint_map.html'
    """
    import folium
    ncs = [
        reproject_netcdf(convert_to_nc(nc).copy(), 'EPSG:4326')  # 'EPSG:3857')
        for nc in ncs
    ]

    # Define the center of the map
    center_lat = np.nanmean(ncs[0].y.to_numpy())  # Latitude
    center_lon = np.nanmean(ncs[0].x.to_numpy())  # Longitude

    # Create a folium map
    m = folium.Map(location=[center_lat, center_lon], 
                   zoom_start=13,
                   control_scale=True)

    for id, nc in enumerate(ncs):
        # Get the footprint data
        footprint_data = np.nanmean(nc.footprint.to_numpy(), axis=0)
        footprint_data = transformer_convention(
            nc, '(lat,lon)', 'top-left', 
            nc.footprint.attrs.get('convention_order', None), 
            nc.footprint.attrs.get('convention_origin', None))(footprint_data)

        # Normalize the footprint data for visualization
        footprint_data_normalized = (footprint_data - np.nanmin(footprint_data)) / \
            (np.nanmax(footprint_data) - np.nanmin(footprint_data))

        # Define the bounds of the footprint data
        bounds = [[np.nanmin(nc.y.to_numpy()), np.nanmin(nc.x.to_numpy())],
                [np.nanmax(nc.y.to_numpy()), np.nanmax(nc.x.to_numpy())]]

        # Name layer
        if labels and len(labels) >= id:
            name = labels[id]
        else:
            name = f'layer {id}'

        # Add the footprint data as an image overlay
        folium.raster_layers.ImageOverlay(
            name=name,
            image=footprint_data_normalized,
            bounds=bounds,
            opacity=0.6,
            colormap=lambda x: get_colormap(id, len(ncs)) + (x,)  # Red colormap
        ).add_to(m)

    if editable:
        return m

    # Add the LayerControl to the map
    folium.LayerControl().add_to(m)
    
    # Save the map as an HTML file
    if dst: 
        m.save(dst)

    # Display the map
    return m


__all__ = [
    "structuredData",
    "convert_to_object",
    "convert_to_nc",
    "convert_to_tif",
    "is_footprint_dict",
    "find_peak",
    "smooth_data",
    "transform_crs",
    "transform_coordinates",
    "find_utm_epsg_from_lon",
    "update_affine",
    "update_attrs_in_nc",
    "fp_to_nc",
    "object_to_nc",
    "attribute_crs",
    "extract_crs",
    "reproject_tif",
    "reproject_netcdf",
    "get_contour_levels",
    "get_contour_vertices",
    "center_footprint",
    "plot_footprint",
    "get_colormap",
    "plot_leaflet",
    "find_middle_point",
    "affine_conventions",
    "transformer_convention",
    "identify_convention",
    "infer_convention_from_affine",
    "infer_convention_from_nc",
]
