# FluxPrint

`FluxPrint` is an open-source Python package implementing flux footprint models
for eddy covariance data analysis. It provides footprint-model implementations
behind a single, consistent interface so researchers can compare spatially
resolved fluxes with field measurements, and add new models by following a small
convention. It is designed for interoperability with ecosystem flux datasets
(e.g. FLUXNET). See Figure 1 for the conceptual scheme.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/conceptual_scheme_dark.png">
  <source media="(prefers-color-scheme: light)" srcset="assets/conceptual_scheme.png">
  <img alt="Conceptual scheme for FluxPrint." src="assets/conceptual_scheme.png" height="200px">
</picture>

*Figure 1. Conceptual scheme for FluxPrint.*

---

## Why FluxPrint?

Most eddy-covariance users already have Kljun's official FFP scripts or
EddyPro's built-in footprint estimates. FluxPrint adds, on top of the same
science:

- **Verified fidelity**: the Kljun et al. (2015) implementation is pinned
  against the original FFP code (vendored in this repository) by regression
  tests asserting **bitwise-identical** fields — a dedicated CI job fails if
  any registered model ever diverges from its reference implementation.
- **A batch workflow**: one call takes a half-hourly table (FLUXNET/ICOS
  style), validates and estimates missing micrometeorology per record, and
  returns a typed `FootprintSeries` ready for climatologies.
- **A typed object API**: georeferencing, source-area contours, plots,
  NetCDF/GeoTIFF/Shapefile export, and provenance metadata on every result.

## Features

- **Footprint models** behind one interface, selected by name (currently
  Kljun et al., 2015; Hsieh et al., 2000 is an experimental work in progress;
  Kormann & Meixner, 2001 is planned).
- **A typed footprint object** (`Footprint`): a 2-D source-area field on a fixed
  grid centred on the tower, plus `FootprintSeries` for time-ordered stacks.
- **Two coordinate frames**: a local, tower-centred metric grid, and a
  georeferenced projected grid (`georeference()`); lon/lat is display-only.
- **Source-area contours**: `contours()` for the 50/80% isopleths,
  `plot()` for a quick look, `to_shapefile()` for GIS.
- **Serialization**: NetCDF is the native format (CF grid mapping included),
  with GeoTIFF conversion; outputs carry provenance (package/model version,
  citation, settings).
- **Aggregation**: collapse a `FootprintSeries` into a climatology.

---

## Installation

```bash
pip install fluxprint
```

Or from source for development:

```bash
pip install git+https://github.com/pedrohenriquecoimbra/fluxprint
```

`import fluxprint` is light: the geo/plotting stack (`xarray`/`netcdf4`,
`rasterio`, `pyproj`, `fiona`/`shapely`, `matplotlib`) is imported lazily by
the operations that need it. Those libraries are still installed as
dependencies today; a future release will move them into the already-declared
extras (`fluxprint[netcdf]`, `[tiff]`, `[crs]`, `[shapefile]`; `[viz]` adds
`folium` for interactive maps, `[all]` installs everything).

---

## Quickstart

### From a met table to a climatology (the batch path)

```python
import pandas as pd
import fluxprint

data = pd.read_csv("halfhours.csv", na_values=[-9999])  # FLUXNET-style table

climatology = fluxprint.wrapper(
    data=data,
    zm=20.0,                        # aerodynamic height: z - d, see note below
    tower=(4321000.0, 3210000.0),   # tower position, for georeferencing
    tower_crs="EPSG:3035",
    dst="climatology.nc",           # optional: write the result
)
```

Columns are matched case-insensitively with common aliases (`USTAR`/`u*`,
`WD`/`wind_direction`, `WS`/`wind_speed`, `OL`/`L`, `SIGMA_V`/`v_sd`, ...).
Missing variables are estimated when physically possible — e.g. the Obukhov
length from `USTAR`/`H`/`TA`/`PA` — and every estimated variable is recorded in
the result's `attrs["estimated_inputs"]`. Crude constant fallbacks require an
explicit `fill_all=True` and warn loudly.

Use `fluxprint.calculate_footprint(data, by="timestamp_column", ...)` for one
footprint per group (a `FootprintSeries`) instead of a single climatology.

> **zm is the aerodynamic height.** All models define `zm = z - d`
> (measurement height above the zero-plane displacement), *not* the instrument
> height. Over vegetation the difference is substantial. If you pass
> `measurement_height` together with `displacement` (or `canopy_height`, using
> d ≈ 0.67·h), FluxPrint derives `zm` for you.

### Compute a single footprint directly

```python
from fluxprint.model import get_model

kljun = get_model("kljun2015")
fp = kljun(
    zm=2.0,          # measurement height above displacement [m]
    z0=0.01,         # roughness length [m]  (or pass umean instead)
    ustar=0.5,       # friction velocity [m s-1]
    pblh=1000.0,     # boundary-layer height [m]
    mo_length=-50.0, # Obukhov length [m]
    v_sigma=0.5,     # std. dev. of lateral velocity [m s-1]
    wind_dir=180.0,  # wind direction [deg] (orients the grid north-up)
    dx=2.0,          # grid spacing [m]
    tower=(4321000.0, 3210000.0),
    tower_crs="EPSG:3035",
)

fp.total()      # fraction of the flux captured by the grid (also in attrs)
fp.peak_xy()    # (x, y) of the footprint peak, metres from the tower
fp.contours([0.5, 0.8])   # 50%/80% source-area isopleths
fp.plot(rs=[0.5, 0.8])    # quick-look plot (matplotlib)
```

Inputs may be scalars (one record) or equal-length sequences (composited into a
single footprint, `fp.n` records). Records failing the model's validity checks
(including NaN gaps) are rejected per record.

### Georeference and export

```python
geo = fp.georeference("EPSG:3035")   # local metres -> projected coords (needs pyproj)
geo.to_netcdf("footprint.nc")        # needs xarray/netcdf4; CF grid mapping included
geo.to_tiff("footprint.tif")         # needs rasterio
geo.to_shapefile("contours.shp", rs=[0.5, 0.8])  # needs fiona
```

---

## Adding a model

A model is a callable mapping micrometeorological inputs to one 2-D `Footprint`
in the local frame. Register it by name and it becomes selectable everywhere:

```python
from fluxprint.model import register_model
from fluxprint.footprint import Footprint

@register_model("my_model", description="My footprint parameterisation")
def calc(*, zm, ustar, pblh, mo_length, v_sigma, wind_dir, z0=None, umean=None,
         domain=None, dx=None, dy=None, tower=None, tower_crs=None, time=None,
         **kwargs) -> Footprint:
    f = ...  # 2-D field on a regular grid centred on the tower
    return Footprint.from_grid(f, dx=dx, tower=tower, tower_crs=tower_crs, time=time)
```

```python
from fluxprint.model import available_models, get_model
available_models()        # ['kljun2015', 'my_model']
get_model("my_model")(...)
```

The registry is backed by `regorator`. New models are expected to come with a
reference oracle in `tests/test_reference_regression.py` pinning them against
their original implementation — CI enforces this.

---

## API reference

### Top level (`import fluxprint`)
- `calculate_footprint(data, by=None, model="kljun2015", ...)` — footprint each
  group of a table; returns a `FootprintSeries`.
- `wrapper(...)` — `calculate_footprint` + aggregation + optional file output.
- `empty_footprint(model, domain=..., dx=...)` — a NaN template on the model's grid.
- `read_handler` / `read_from_url` / `read_from_file` — table/NetCDF/TIFF readers.
- Legacy `write_to_*` helpers are deprecated: use the `Footprint.to_*` methods.

### `fluxprint.model`
- `get_model(name)` — return the registered model callable.
- `available_models()` — list registered model names.
- `register_model(name, description="", **attrs)` — decorator to register a model.
- `FootprintModel` — the callable protocol models conform to.

### `fluxprint.footprint.Footprint`
- `Footprint.from_grid(f, dx, dy=None, **meta)` — build a local, tower-centred footprint.
- `georeference(target_crs)` / `to_lonlat()` — local → projected; display lon/lat.
- `total()`, `peak_xy()`, `normalized()`, `contours(rs)` — analysis helpers.
- `plot(rs=None)` — matplotlib quick look (never calls `plt.show()`).
- `to_netcdf` / `from_netcdf`, `to_tiff` / `from_tiff`, `to_xarray` /
  `from_xarray`, `to_shapefile`.

### `fluxprint.footprint.FootprintSeries`
- `aggregate(smooth=True)` — collapse the stack to a 2-D climatology.
- `georeference(target_crs)`, `to_netcdf` / `from_netcdf`, `to_xarray` / `from_xarray`.

See [CHANGELOG.md](CHANGELOG.md) for release notes and deprecations.

---

## Examples

See the `sample/` directory for usage examples.

---

## Citing FluxPrint

If FluxPrint contributes to your research, please cite the software (see
[CITATION.cff](CITATION.cff)) alongside the footprint-model paper you used
(e.g. Kljun et al., 2015 — the models' own citations and DOIs are stamped
into every output's metadata).

---

## Contributing

Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md) for the
development setup, and open a pull request from a feature branch.

---

## License

Licensed under the European Union Public Licence v. 1.2 (EUPL-1.2). See the
[LICENSE](LICENSE) file for details. The vendored Kljun et al. (2015) FFP
reference code (`fluxprint/model/Kljun_et_al_2015_original/`) is distributed
under its own permissive licence (see the `license.txt` alongside it),
copyright (C) 2015–2023 Natascha Kljun.

---

## Acknowledgments

- Kljun, N., Calanca, P., Rotach, M. W., Schmid, H. P. (2015): *The simple
  two-dimensional parameterisation for Flux Footprint Predictions (FFP)*,
  Geosci. Model Dev. 8, 3695–3713, doi:10.5194/gmd-8-3695-2015.
- Hsieh, C.-I., Katul, G., Chi, T. (2000): *An approximate analytical model for
  footprint estimation of scalar fluxes in thermally stratified atmospheric
  flows*, Adv. Water Resour. 23, 765–772, doi:10.1016/S0309-1708(99)00042-1.
- Kormann, R., Meixner, F. X. (2001): *An analytical footprint model for
  non-neutral stratification*, Boundary-Layer Meteorol. 99, 207–224,
  doi:10.1023/A:1018991015119.

---

## Contact

- [Pedro Henrique Coimbra](mailto:pedro-henrique.herig-coimbra@inrae.fr)
- [GitHub Issues](https://github.com/pedrohenriquecoimbra/fluxprint/issues)
