import warnings

import numpy as np

_DEFAULT_ATTRS = {
    '__global__': {
        # f'Diurnal Footprints for {data_to_nc.sitename} at 30-min resolution',
        'Title': 'Flux Footprint',
        'Creation_Date': "",  # datetime.datetime.now().strftime('%d-%b-%Y'),
        'Contact': 'Pedro Coimbra and Benjamin Loubet at ECOSYS, INRAE, AgroParisTech, Université Paris-Saclay, Palaiseau, France, pedro-henrique.herig-coimbra@inrae.fr and benjamin.loubet@inrae.fr',
        'Aknowledgement': 'Similar work has been done by Betty Molinier and Natascha Kljun, Centre for Environmental and Climate Science, Lund University, betty.molinier@cec.lu.se and natascha.kljun@cec.lu.se',
        'Conventions': 'CF-1.8',
        # 'Creator': 'Betty Molinier¹ (ORCID: 0000-0002-7212-4120), Natascha Kljun¹ (ORCID: 0000-0001-9650-2184), Pedro Coimbra² (ORCID: 0009-0008-6845-8735) and Benjamin Loubet² (ORCID: 0000-0001-8825-8775).\n1 Centre for Environmental and Climate Science, Lund University, Sweden.\n2 ECOSYS, INRAE, AgroParisTech, Université Paris-Saclay, Palaiseau, France',
        # 'Institution': 'Centre for Environmental and Climate Science, Lund University, Lund, Sweden\nECOSYS, INRAE, AgroParisTech, Université Paris-Saclay, Palaiseau, France',
        'Source': "",  # 'Ecosystem Thematic Centre (2024). ETC NRT Fluxes, Romainville, 2023-02-13–2024-04-30, ICOS Cities, https://hdl.handle.net/11676/ML3hTCCg5neiu2yw_HUF7AkW; Hersbach, H., et al. (2023): ERA5 hourly data on single levels from 1940 to present. Copernicus Climate Change Service (C3S) Climate Data Store (CDS), DOI: 10.24381/cds.adbb2d47 (Accessed on 05-May-2024)',
        'Model_Used': 'FFP, Kljun et al. (2015), doi:10.5194/gmd‐8‐3695‐2015',
        # f'This file contains flux footprints for the {data_to_nc.sitename} flux tower in France at 30-minute temporal resolution. The name of the file includes the date of all footprints contained.',
        'Summary': "",
        'Subjects': "",  # 'Flux footprints, atmospheric modelling, urban flux, ICOS Cities',
        'Coordinate_Reference_System': "",  # 'WGS 84',
        # '+proj=tmerc +lat_0=48.88514 +lon_0=2.42222 +k=1 +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m +no_defs',
        'crs_projection4': "",
        'crs_wkt': "",  # 'PROJCS["WGS_1984_Transverse_Mercator",GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["Degree",0.0174532925199433]],PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",48.885140],PARAMETER["central_meridian",2.422220],PARAMETER["scale_factor",1],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["Meter",1]]',
        'Variables': 'Time, X, Y, Boundary Layer Height Quality Flag, Footprint Climatology',
        'Tower_Location_Latitude': np.nan,
        'Tower_Location_Longitude': np.nan,
        'Tower_Location_CRS': "",
        'Tower Height (m)': np.nan,
        'Frequency': '',  # '30 min'
    },
    'timestep': {'units': 'yymmddhhMM', 'timezone': ''},
    'x': {'long_name': 'x coordinate of projection',
                       'standard_name': 'projection_x_coordinate',
                       'units': 'meters'},
    'y': {'long_name': 'y coordinate of projection',
                       'standard_name': 'projection_y_coordinate',
                       'units': 'meters'},
    'footprint': {'long_name': "footprint",
                  "units": "per square meter",
                  "10^?": 0},
}


def __getattr__(name):
    # PEP 562 deprecation shim: the public name warns, the private one is the
    # internal spelling (used by fluxprint.utils without triggering this).
    if name == "DEFAULT_ATTRS":
        warnings.warn(
            "fluxprint.template.DEFAULT_ATTRS is deprecated and will be "
            "removed in a future release; provenance attrs are stamped by "
            "the models, and empty_footprint() is the template API.",
            DeprecationWarning, stacklevel=2)
        return _DEFAULT_ATTRS
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
