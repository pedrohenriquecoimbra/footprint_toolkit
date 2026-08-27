# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project aims to follow [Semantic Versioning](https://semver.org/)
(pre-1.0: minor releases may contain breaking changes, announced here).

## [0.4.0] - in progress

The genericization release: a model is only physics. The scaffolding every
footprint model used to copy now lives in a generic layer, and the registered
Kljun model remains bitwise identical to the vendored reference at every
step (CI-enforced).

### Added
- `fluxprint.grid`: `GridSpec` (output shape/axes known without compute),
  `resolve_grid()` (the FFP domain/dx/dy/nx/ny reconciliation rules, hoisted
  verbatim), `GridContext` (cached cartesian + lazy polar coordinates), and
  the wind-frame primitives `rotate_theta()` / `to_wind_frame()` with a
  documented meteorological convention.
- `fluxprint.model.engine`: the model-agnostic driver — `normalize_inputs()`,
  `run_climatology()` (validate/accumulate/normalize/smooth loop with
  reference-exact `flag_err` bookkeeping), `build_footprint()` (provenance
  attrs + `captured_fraction` diagnostic), and the `@footprint_model`
  decorator that turns a bare per-record kernel into a fully registered
  model with the canonical signature.
- The Kljun port now runs on this pipeline; its physics is the module-private
  `_kljun_record` kernel (moved verbatim, bug-for-bug). The file shrank by
  ~200 lines of scaffolding.
- `smooth` as the generic spelling of the model-level smoothing knob;
  `smooth_data` remains fully supported (downstream-pinned). `smooth` wins
  when both are given.
- `empty_footprint()` resolves the grid directly via the model's
  `resolve_grid` hook — no more placeholder model run (microseconds instead
  of milliseconds; grid equality with a real run is pinned by a test).
- Regression safety net: grid-variant oracle cases (default/nx-only/dx-only/
  domain+nx/dx+nx) and a dtype pin in the reference-equivalence suite; a
  downstream-contract test file pinning the surface fluxcom's FFPProvider
  depends on.
- `map_footprints()`: the xarray/dask-native compute path. Maps the model's
  per-record kernel over arrays of any dimensionality (aliases recognized)
  and returns `(*input_dims, y, x)`; lazy over dask-backed inputs; pinned to
  reproduce single-record model calls bitwise. Per-record validation is
  silent on this path — rejections are summarised once per block instead of
  logged one line per record.
- `Footprint.weighted_mean(field, min_coverage=None)`: footprint-weighted
  mean of a gridded quantity owning the NaN mask and coverage threshold; a
  uniform field of 1.0 returns exactly 1.0 regardless of normalization or
  domain truncation.
- `calculate_footprint(on_error=...)`: `"skip"` (default, previous
  behavior), `"raise"`, or `"nan"` — an all-NaN member per failed group with
  the reason in `attrs["error"]`, so long batches keep one slot per group.
- The kernel protocol on model callables: `.kernel`, `.resolve_grid`,
  `.validate`, `.model_options`, `.option_defaults` — attached by
  `@footprint_model` and by the Kljun adapter; `map_footprints` and
  `empty_footprint` are built on it.

### Removed
- The unreachable `crop` block in `calc_ffp_climatology` (never reachable
  through the package API). Passing `crop`/`rs` now emits a
  `DeprecationWarning` pointing at `Footprint.contours()`/`level_for()`.

## [0.3.1] - unreleased

Additive generic-layer API (the 0.3.1 fast-follow); no model files touched.

### Added
- `fluxprint.footprint.smooth_field()` and `FFP_SMOOTH_KERNEL`: the standard
  FFP 3x3 double-convolution smoothing as a model-agnostic primitive, and
  `Footprint.smoothed()` to apply it to any footprint.
  `FootprintSeries.aggregate` and the deprecated `aggregate_footprints` now
  share this single implementation.
- `Footprint.level_for(r)`: the source-area field level for a fraction `r`,
  without contour extraction; `Footprint.contours()` uses the same search.
  Fractions are of the full (unit-integral) model footprint, not of the
  captured total - the two differ on any truncated domain.
- `Footprint.captured_fraction`: live property (equal to `total()`), the
  generic counterpart of the model-stamped `attrs["captured_fraction"]`.
- `fluxprint.ALIASES`: the exported table of column aliases recognized by
  `process_footprint_inputs` (model inputs and estimator drivers), so
  downstream integrations no longer maintain their own copy.

### Changed
- `process_footprint_inputs(data=...)` now raises a `TypeError` naming the
  supported containers when `data` is neither a DataFrame nor a dict (e.g. an
  `xarray.Dataset`); previously such input was silently ignored and the
  computation proceeded from kwargs alone.

### Deprecated
- `fluxprint.utils.smooth_data()`: use `fluxprint.footprint.smooth_field()`.

## [0.3.0] - unreleased

### Added
- Provenance on every model output: `attrs` now carry the fluxprint version,
  the model's citation/DOI/reference version, the wind-profile input mode,
  smoothing/rslayer settings and a `history` line; `register_model` accepts
  citation metadata.
- CF metadata on NetCDF output: a `Conventions` attribute and, for
  georeferenced footprints, a CF `grid_mapping` variable (`spatial_ref`) so
  GDAL/QGIS/rioxarray recognize the CRS.
- Displacement-height support: pass `measurement_height` with `displacement`
  (or `canopy_height`, d ~ 0.67 h) and `zm = z - d` is derived; matching
  column aliases are recognized in tables.
- Optional extras declared: `fluxprint[netcdf]`, `[tiff]`, `[crs]`,
  `[shapefile]`, `[viz]` (folium), `[all]`, `[dev]` - making the package's
  ImportError hints valid. The packages remain core dependencies for now.
- `netcdf4` added as a dependency (the `.nc` read path always required it).
- CI: a GitHub Actions matrix (3.10-3.13) plus a dedicated
  `reference-equivalence` job asserting every registered model reproduces its
  vendored reference implementation exactly; the INRAE mirror now runs only
  after both pass. An import-lightness check keeps `import fluxprint` free of
  the geo stack.
- `CITATION.cff` and `CONTRIBUTING.md`.
- `Footprint.contours(rs)`: source-area isopleths (e.g. the 50/80% contours)
  computed directly on the footprint object, with per-contour level, enclosed
  fraction, vertices, and an open/closed flag.
- `Footprint.plot(rs=...)`: quick-look plot of the field with optional
  isopleth overlay (matplotlib; never calls `plt.show()`).
- `Footprint.to_shapefile(path, rs=...)`: export closed isopleths as polygons
  (fiona), carrying the CRS when georeferenced.
- Golden regression tests pinning the Kljun port to the vendored reference
  implementation across stability regimes, plus a pinned crosswind-integrated
  peak distance.
- GitHub Actions test workflow (Python 3.10-3.13).
- `attrs["captured_fraction"]` on every Kljun footprint, with a warning when
  the domain truncates more than 20% of the flux.
- Estimated input variables are recorded in `attrs["estimated_inputs"]` and
  survive aggregation; `process_footprint_inputs` exposes them via the
  returned mapping's `.estimated` attribute.
- `FluxPrintError` / `InputValidationError` exception hierarchy.

### Changed
- `import fluxprint` no longer loads the geo/plotting stack (rasterio,
  xarray, pyproj, fiona, shapely, matplotlib): those imports are deferred to
  the operations that need them; `fluxprint.utils` loads lazily on first use.
- tz-aware timestamps are converted to naive UTC with an explicit warning;
  fluxprint stores times as naive timestamps (flux networks conventionally
  use local standard time - document your convention).
- The vendored Kljun reference code's license notice now ships in the wheel
  and sdist (PEP 639 `license-files`); `regorator` is version-pinned; the
  smoothing kernel uses `np.array` instead of the deprecated `np.matrix`.
- **Crude constant fills are now opt-in**: `fill_all` defaults to `False`, the
  `pblh=1000` constant moved to the crude tier, and every crude fill emits a
  `UserWarning`. Missing `wind_dir`/`zm`/`pblh` now raise a clear error
  instead of being silently fabricated.
- **Batch smoothing matches the reference FFP procedure**: `calculate_footprint`
  computes members with `smooth_data=0` and `FootprintSeries.aggregate()`
  smooths the climatology once (previously both were smoothed - four kernel
  passes instead of two). Direct model calls keep `smooth_data=1`.
- `Footprint.normalized()` divides by `total()` so the result *integrates*
  (not sums) to one, honouring its documented contract.
- `wrapper` stamps the resolved model name under `attrs["model"]` (the
  duplicate `model_used` key is gone) and `aggregate()` carries attrs shared
  by all members onto the climatology.
- Fatal FFP validation errors raise `InputValidationError` with their full
  message at every verbosity (previously `Exception('')` at the default);
  error codes log at warning level, routine alerts at info, instead of
  `print()`.
- FLUXNET-style tables now work end to end: `H`/`TA`/`PA` driver columns are
  extracted so the Obukhov-length estimator can run, `mo_length` is estimated
  before `z0`, and `L`/`wind_speed`/`sigma_v`/`wind_direction` aliases are
  recognized. `z0`-only tables are accepted without `umean`.

### Fixed
- NaN met records passed validation and silently zeroed or NaN-poisoned
  composited footprints; they are now rejected per record (new code 21).
- `smooth_data=0` never actually disabled smoothing (the check was
  `is not None`); fixed in the Kljun port, `aggregate_footprints`, and the
  Hsieh draft.
- The z0 estimator defaulted a missing Obukhov length to 1 m (extreme
  stability), producing astronomically wrong roughness lengths; it now
  requires a real `mo_length`, and `compute_mo_length` clamps the H=0 neutral
  limit to a finite |L| instead of returning inf.
- The documented by-timestamp batch workflow crashed `to_netcdf` (raw
  `pd.Timestamp` group labels in attrs); attrs are now coerced to
  NetCDF-safe types.
- `calc_footprint_1d`'s invalid-input branches crashed with `TypeError`
  (unpacking `None`) and its validation used misaligned exception codes that
  logged instead of aborting; it now validates strictly (including NaN) and
  returns a clean `flag_err=1` result for the no-solution case.

### Deprecated
- `fluxprint.get_contour` (use `Footprint.contours()`),
  `fluxprint.aggregate_footprints` (use `FootprintSeries.aggregate()`), and
  the legacy `io.write_to_file`/`write_to_netcdf`/`write_to_shapefile`/
  `write_to_raster` writers (use the corresponding `Footprint.to_*` methods).
  The writers now accept `Footprint`/`FootprintSeries` objects and delegate,
  with a `DeprecationWarning`; they will be removed in a future release.

### Removed
- `fluxprint/ext_libs/`: a byte-identical duplicate of the vendored Kljun
  reference, a committed binary wheel and two stale notebooks - none of it
  reachable from the package.
- The `Kormann_and_Meixner_2001` module: it was untranslated scaffolding that
  raised `NameError` on any call. The model remains planned; the reference
  math will be ported against the paper in a future release.
- The ported single-footprint `FFP()` function in the Kljun module: it
  crashed with `UnboundLocalError` on any `crop=False` call and used
  misaligned exception codes. Use the registered `calc()` (one record) or the
  vendored reference implementation instead.

## [0.2.0] - 2026-07-06

### Fixed
- Micrometeorology estimators reconciled with their references: the Obukhov
  length now divides the sensible heat flux by rho*cp (it was ~1200x too
  small), and the boundary-layer height estimate stays finite at the equator.

## [0.1.0] - 2026-06-03

### Changed
- **Breaking**: `core.calculate_footprint` returns a `FootprintSeries` (of
  `Footprint` value objects) instead of the previous dict-style output;
  models are resolved through a registry (`fluxprint.model.get_model`).

Earlier development snapshots (0.0.2-0.0.4, 2025) predate this changelog.
