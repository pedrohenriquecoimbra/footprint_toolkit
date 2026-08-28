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
- **`hsieh2000` is a registered model**: Hsieh et al. (2000) with the Detto
  et al. (2006) crosswind expansion, implemented as an analytic kernel on
  the generic driver (`fluxprint/model/Hsieh_et_al_2000.py` shrank to
  equations + registration; `peak_distance()` exposes Eq 19). The pre-0.4
  experimental draft — never registered — is replaced: it rotated the grid
  with a math-angle rotation and returned per-pixel fractions; the
  registered model evaluates on the fixed grid via the wind-frame transform
  and returns a proper m**-2 density. `z0` is required (no `umean` mode);
  `pblh` participates only in validation. Pinned by a golden-snapshot
  oracle (`tests/data/hsieh2000_reference.npz`, regenerate script alongside)
  plus paper-derived invariants (cumulative `exp(-A/x)`, Eq 19 peak
  distance, Detto Eq B4 crosswind spread) in `tests/test_hsieh_reference.py`.
  The draft's `patch_index`/`patch_ffp` helpers are gone — superseded by the
  generic `Footprint.weighted_mean()`.

### Added
- `FootprintSeries` is a real container now: slicing returns a
  `FootprintSeries` (previously a bare list), `append()` grows a series with
  the shared-grid/shared-frame validation applied (direct
  `series.footprints.append` bypasses it), and the series has an
  informative `repr`.
- `Footprint.replace(**changes)`: the public variant constructor (copies
  `attrs`; shares arrays unless replaced). `_replace` remains as the
  internal spelling.

### Fixed
- User attrs that shadow reserved frame metadata (`crs`, `crs_wkt`,
  `crs_proj4`, `tower_x`, `tower_y`, `tower_crs`, `n_records`) now warn and
  are dropped at serialization instead of overriding the real frame — a
  note like `attrs["crs"] = "WGS84"` could previously become the
  authoritative CRS on reload. `crs_wkt`/`crs_proj4` also no longer
  reappear as user attrs after a round-trip.
- A series with no per-member time labels (e.g. grouped climatologies from
  `calculate_footprint(by=<categorical>)`) now writes a *marked* index time
  axis, and `from_xarray`/`from_netcdf` restore `time=None` instead of
  promoting the index to fake relative labels. This also fixes the
  write-succeeds/read-fails asymmetry: a georeferenced climatology series
  NetCDF is now readable by the library that wrote it. Mixed per-member
  labels degrade to the marked index with an explicit warning.
- `calculate_footprint(on_error="nan")` gives every failed group its own
  field array; previously all NaN slots in a series shared one buffer, so
  editing one member in place silently rewrote the others.
- `FootprintSeries.aggregate(smooth=True)` now stamps `attrs["smoothed"]=1`
  and drops the members' `smooth_data`/`smoothed` attrs instead of carrying
  `smooth_data=0` onto a field it just smoothed — which invited a second,
  footprint-widening smoothing pass downstream.

### Changed
- The `regorator` upper bound (`<0.3`) is dropped: fluxprint uses a
  two-function surface that is stable across regorator releases, and an
  upper cap on a library dependency only creates resolver conflicts
  downstream. The floor stays at `>=0.2` (the version CI exercises).
- `Footprint` construction now rejects descending or empty coordinate axes
  with an actionable error (flip north-first rasters before constructing);
  previously a descending axis silently flipped the sign of `dx`/`dy` and
  every cell-area-dependent quantity (`total()`, coverage, contour levels).
- `calculate_footprint`'s docs (and the README) now state the eager-series
  vs `map_footprints` division of labor: series for grouped climatologies,
  the mapped path for one-footprint-per-record at scale.
- `FootprintSeries.aggregate()` computes the climatology incrementally (two
  grid-plane accumulators, one pass) instead of materializing the full
  `(nt, ny, nx)` stack plus nanmean's internal copies — peak memory for the
  mean is now two planes regardless of series length. Same nan-mean
  semantics (per-cell NaN holes ignored; all-NaN cells stay NaN).
- The registered `kljun2015` adapter is now generated by `@footprint_model`
  instead of hand-written — the model file no longer duplicates the generic
  driver line for line. Same signature, same numerics (bitwise-pinned), same
  provenance attrs; only the function's `__name__` changes to
  `kljun2015_calc`.
- Column-alias resolution in `process_footprint_inputs` is now one
  first-match-wins scan (canonical name, then the `ALIASES` spellings in
  declared order), matching `map_footprints`. Two edge cases change: with
  two different alias columns present (e.g. `wd` and `wind_direction`) the
  first declared alias now wins instead of the last, and a
  case-insensitively matched canonical column (e.g. `WIND_DIR`) now beats an
  exact alias column. Resolved values are consistently lists.

### Deprecated
- `fluxprint.utils.get_contour_levels()` / `get_contour_vertices()`: use
  `Footprint.level_for()` / `Footprint.contours()`.
- `fluxprint.template.DEFAULT_ATTRS` (PEP 562 warning on access): provenance
  attrs are stamped by the models; `empty_footprint()` is the template API.
- The remaining zero-caller legacy helpers in `fluxprint.utils` now warn:
  `structuredData`, `transform_crs` (use `transform_coordinates`),
  `center_footprint`/`update_affine` (use `Footprint.georeference()`),
  `plot_footprint` (use `Footprint.plot()`), `is_footprint_dict`,
  `find_utm_epsg_from_lon`, `find_middle_point`, `identify_convention`,
  `attribute_crs`, `reproject_tif`.

### Removed
- The unreachable `crop` block in `calc_ffp_climatology` (never reachable
  through the package API). Passing `crop`/`rs` now emits a
  `DeprecationWarning` pointing at `Footprint.contours()`/`level_for()`.
- `fluxprint.aggregate_footprints` (deprecated in 0.3.0; one full cycle
  served): use `FootprintSeries.aggregate()`. The removal of the also-
  deprecated `get_contour` is deferred to the release that removes the
  legacy `io.write_*` writers, whose shapefile writer still calls it.
- Dead code: an unreachable duplicate block in `utils.convert_to_object`, the
  broken `utils.find_utm_epsg_from_lon_deprecated`, the unused
  `commons.ensure_supported_dtype` machinery, the unused
  `fluxprint.model.kljun2015_o` alias (the vendored reference stays
  importable by its full path, and `import fluxprint` no longer loads it),
  and leftover unused imports.

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
