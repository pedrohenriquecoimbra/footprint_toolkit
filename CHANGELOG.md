# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project aims to follow [Semantic Versioning](https://semver.org/)
(pre-1.0: minor releases may contain breaking changes, announced here).

## [Unreleased]

### Added
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
