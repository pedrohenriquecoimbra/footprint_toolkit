"""Footprint value objects.

A :class:`Footprint` is the 2-D source-area weight field of a flux measurement
on a fixed, regular grid. A :class:`FootprintSeries` is a time-ordered stack of
such fields ``(time, y, x)`` sharing one grid; aggregating a series yields a 2-D
climatology.

**Two spatial frames, keyed by ``crs``:**

* ``crs is None`` -- the *local* frame the models produce: ``x``/``y`` are
  metres relative to the tower, which sits at the origin ``(0, 0)``.
* ``crs`` set -- a *georeferenced* frame: ``x``/``y`` are real projected
  coordinates (a metric CRS such as ``EPSG:3035``). Longitude/latitude is
  treated as display-only (:meth:`Footprint.to_lonlat`), never as the grid,
  so the grid stays regular.

**Frame invariant:** a georeferenced footprint must not carry a *relative* time
label -- its ``time`` is a real :class:`datetime.datetime` or ``None``.

The in-memory objects depend only on :mod:`numpy`. NetCDF is the native
serialization format and TIFF a supported conversion; ``xarray``/``netcdf4``,
``rasterio`` and ``pyproj`` are optional and imported lazily.
"""
from __future__ import annotations

import importlib
import logging
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Iterator

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    import xarray as xr

logger = logging.getLogger("fluxprint.footprint")

__all__ = ["Footprint", "FootprintSeries", "laea_crs", "smooth_field",
           "FFP_SMOOTH_KERNEL"]

#: The standard FFP 3x3 smoothing kernel (Kljun et al. 2015). One smoothing
#: pass is one 2-D convolution with this kernel; the reference procedure
#: applies it twice.
FFP_SMOOTH_KERNEL = np.array([[0.05, 0.1, 0.05],
                              [0.10, 0.4, 0.10],
                              [0.05, 0.1, 0.05]])


def smooth_field(f: np.ndarray, kernel: np.ndarray | None = None,
                 passes: int = 2) -> np.ndarray:
    """Smooth a 2-D footprint field with the standard FFP kernel.

    This is the model-agnostic smoothing step of the reference FFP procedure:
    ``passes`` 2-D convolutions (``mode="same"``) with the 3x3
    :data:`FFP_SMOOTH_KERNEL`. It applies to any footprint field, not just
    Kljun's. Requires ``scipy``.

    Args:
        f: 2-D field to smooth.
        kernel: Alternative convolution kernel; defaults to
            :data:`FFP_SMOOTH_KERNEL`.
        passes: Number of convolution passes (the FFP reference uses 2).

    Returns:
        The smoothed field (a new array; ``f`` is not modified).
    """
    sg = _require("scipy.signal", "")  # scipy is a core dependency
    kernel = FFP_SMOOTH_KERNEL if kernel is None else np.asarray(kernel)
    out = np.asarray(f)
    for _ in range(passes):
        out = sg.convolve2d(out, kernel, mode="same")
    return out


def _require(module: str, extra: str) -> Any:
    """Import an optional dependency or raise a clear, actionable error."""
    try:
        return importlib.import_module(module)
    except ImportError as exc:  # pragma: no cover - exercised via public methods
        hint = f" Install it with: pip install fluxprint[{extra}]" if extra else ""
        raise ImportError(f"{module!r} is required for this operation.{hint}") from exc


def _is_regular(coord: np.ndarray, *, rtol: float = 1e-3) -> bool:
    """Return True if a 1-D coordinate is evenly spaced (within ``rtol``)."""
    if coord.size < 2:
        return True
    steps = np.diff(coord)
    return bool(np.allclose(steps, steps[0], rtol=rtol))


def _ring_signed_area(ring: np.ndarray) -> float:
    """Shoelace signed area of a closed ring (positive = counter-clockwise)."""
    x, y = ring[:, 0], ring[:, 1]
    return float(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]) / 2.0)


def _point_in_ring(point: np.ndarray, ring: np.ndarray) -> bool:
    """Ray-casting point-in-polygon test against a closed ring."""
    x, y = float(point[0]), float(point[1])
    vx, vy = ring[:-1, 0], ring[:-1, 1]
    wx, wy = ring[1:, 0], ring[1:, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        crossing = ((vy > y) != (wy > y)) & (
            x < (wx - vx) * (y - vy) / (wy - vy) + vx)
    return bool(np.count_nonzero(crossing) % 2)


def _nest_rings(segments: list[np.ndarray]) -> list[list[list[tuple]]]:
    """Group closed rings into GeoJSON-style polygons ``[exterior, holes...]``.

    Containment depth decides the role: even depth = the exterior of a
    polygon, odd depth = a hole of its smallest enclosing exterior. Annular
    level sets (e.g. climatologies over opposing wind sectors) produce holes;
    writing every ring as a solid polygon would double-count them. Exteriors
    are oriented counter-clockwise, holes clockwise.
    """
    rings = []
    for seg in segments:
        ring = np.asarray(seg, dtype=float)
        if not np.array_equal(ring[0], ring[-1]):
            ring = np.vstack([ring, ring[:1]])
        rings.append(ring)
    n = len(rings)
    contains = [[j != i and _point_in_ring(rings[i][0], rings[j])
                 for j in range(n)] for i in range(n)]
    depth = [sum(row) for row in contains]
    areas = [abs(_ring_signed_area(ring)) for ring in rings]

    def _as_coords(ring: np.ndarray, *, clockwise: bool) -> list[tuple]:
        if (_ring_signed_area(ring) < 0) != clockwise:
            ring = ring[::-1]
        return [(float(px), float(py)) for px, py in ring]

    polygons = {i: [_as_coords(rings[i], clockwise=False)]
                for i in range(n) if depth[i] % 2 == 0}
    for i in range(n):
        if depth[i] % 2 == 1:
            parent = min((j for j in polygons if contains[i][j]),
                         key=lambda j: areas[j])
            polygons[parent].append(_as_coords(rings[i], clockwise=True))
    return list(polygons.values())


def _is_relative_label(time: Any) -> bool:
    """True if ``time`` is a numeric (relative) label rather than a timestamp."""
    return isinstance(time, (int, float)) and not isinstance(time, bool)


def laea_crs(lat: float, lon: float, ellps: str = "GRS80") -> str:
    """Return a Lambert Azimuthal Equal-Area CRS (proj4) centred at ``(lat, lon)``.

    Equal-area and centred on the tower, so a tower-centred footprint grid maps
    onto it with the tower at the projection origin (local metres == projected
    metres). Usable directly as a CRS string (pyproj/rasterio) and as the
    default target of :meth:`Footprint.georeference`.
    """
    return (f"+proj=laea +lat_0={lat} +lon_0={lon} "
            f"+ellps={ellps} +units=m +no_defs")


def _tower_laea(pyproj, tower, tower_crs) -> str:
    """Tower-centred LAEA CRS from the tower position expressed in ``tower_crs``."""
    to_geo = pyproj.Transformer.from_crs(tower_crs, "EPSG:4326", always_xy=True)
    lon, lat = to_geo.transform(tower[0], tower[1])
    return laea_crs(lat, lon)


@dataclass(slots=True, eq=False)
class Footprint:
    """A flux footprint: one 2-D source-area weight field on a fixed grid.

    The grid is regular. In the local frame (``crs is None``) it is centred on
    the tower at ``(0, 0)`` with ``x`` increasing eastward and ``y`` northward,
    both in metres. In the georeferenced frame (``crs`` set) ``x``/``y`` are
    real projected coordinates. The field is a density in m\\ :sup:`-2`.

    Attributes:
        f: Footprint weight field, shape ``(ny, nx)``, units m**-2.
        x: 1-D west-east cell centres, length ``nx`` (metres from tower if
            ``crs is None``, else real projected coordinates).
        y: 1-D south-north cell centres, length ``ny``.
        time: Real timestamp for a single interval, a numeric label
            (e.g. hour-of-day; local frame only), or ``None`` for a climatology.
        crs: ``None`` for the local tower-centred frame, else the projected CRS
            of ``x``/``y`` (e.g. ``"EPSG:3035"``).
        tower: ``(x, y)`` of the tower in ``tower_crs``; needed to georeference.
        tower_crs: CRS of ``tower``.
        n: Number of valid input records aggregated into ``f`` (climatology).
        attrs: Free-form, CF-style metadata carried through to NetCDF.
    """

    f: np.ndarray
    x: np.ndarray
    y: np.ndarray
    time: datetime | float | None = None
    crs: str | None = None
    tower: tuple[float, float] | None = None
    tower_crs: str | None = None
    n: int | None = None
    attrs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.f = np.asarray(self.f, dtype=float)
        self.x = np.asarray(self.x, dtype=float)
        self.y = np.asarray(self.y, dtype=float)

        if self.f.ndim != 2:
            raise ValueError(f"f must be 2-D (ny, nx); got shape {self.f.shape}.")
        if self.f.shape != (self.y.size, self.x.size):
            raise ValueError(
                f"f has shape {self.f.shape} but coordinates imply "
                f"{(self.y.size, self.x.size)} (ny, nx).")
        if not (_is_regular(self.x) and _is_regular(self.y)):
            raise ValueError("x and y must be regularly spaced.")
        if self.crs is not None and _is_relative_label(self.time):
            raise ValueError(
                "a georeferenced footprint (crs set) cannot carry a relative "
                "time label; use a timestamp (datetime) or None.")
        if isinstance(self.time, datetime) and self.time.tzinfo is not None:
            # Times are stored naive: serialization has no timezone
            # representation, so a tz-aware stamp would be coerced silently
            # later. Convert explicitly (to UTC) and say so. Flux networks
            # conventionally use *local standard time* - document which
            # convention your timestamps follow.
            warnings.warn(
                "tz-aware timestamp converted to naive UTC "
                f"({self.time.isoformat()}); fluxprint stores times as naive "
                "timestamps.", UserWarning, stacklevel=3)
            self.time = self.time.astimezone(timezone.utc).replace(tzinfo=None)

    # -- geometry ---------------------------------------------------------- #
    @property
    def nx(self) -> int:
        """Number of grid columns (west-east)."""
        return self.x.size

    @property
    def ny(self) -> int:
        """Number of grid rows (south-north)."""
        return self.y.size

    @property
    def dx(self) -> float:
        """West-east grid spacing."""
        return float(self.x[1] - self.x[0]) if self.nx > 1 else 0.0

    @property
    def dy(self) -> float:
        """South-north grid spacing."""
        return float(self.y[1] - self.y[0]) if self.ny > 1 else 0.0

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """Cell-centre bounds ``(xmin, xmax, ymin, ymax)``."""
        return (float(self.x.min()), float(self.x.max()),
                float(self.y.min()), float(self.y.max()))

    @property
    def is_climatology(self) -> bool:
        """True if this footprint has no single timestamp (time-aggregated)."""
        return self.time is None

    @property
    def is_georeferenced(self) -> bool:
        """True if ``x``/``y`` are real coordinates in a projected ``crs``."""
        return self.crs is not None

    def meshgrid(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the 2-D ``(x, y)`` coordinate grids matching ``f``."""
        return np.meshgrid(self.x, self.y)

    # -- analysis ---------------------------------------------------------- #
    def total(self) -> float:
        """Integral of the field over the domain (``sum(f) * dx * dy``)."""
        return float(np.nansum(self.f) * self.dx * self.dy)

    def peak_xy(self) -> tuple[float, float]:
        """``(x, y)`` of the maximum footprint weight."""
        iy, ix = np.unravel_index(np.nanargmax(self.f), self.f.shape)
        return float(self.x[ix]), float(self.y[iy])

    def normalized(self) -> "Footprint":
        """Return a copy whose field integrates to one over the domain.

        The field is divided by :meth:`total` (``sum(f) * dx * dy``), so the
        result is a density in ``m**-2`` whose *integral* -- not its sum -- is
        one.
        """
        scale = self.total()
        if scale == 0:
            raise ValueError(
                "Cannot normalize a footprint whose integral is zero (empty "
                "field, or a degenerate single-row/column grid).")
        return self._replace(f=self.f / scale)

    def smoothed(self, kernel: np.ndarray | None = None,
                 passes: int = 2) -> "Footprint":
        """Return a copy smoothed with the standard FFP kernel.

        Applies :func:`smooth_field` (``passes`` convolutions with the 3x3
        :data:`FFP_SMOOTH_KERNEL`) to the field — the same smoothing the
        reference FFP procedure applies to a climatology, usable with any
        footprint model. The copy records ``attrs["smoothed"] = 1``.
        """
        out = self._replace(f=smooth_field(self.f, kernel=kernel,
                                           passes=passes))
        out.attrs["smoothed"] = 1
        return out

    def contours(self, rs: Any = (0.5, 0.8)) -> list[dict]:
        """Source-area isopleths: the contour enclosing ``r`` of the flux.

        For each requested fraction ``r``, finds the field level whose
        enclosed integral is nearest to ``r`` (the full model footprint
        integrates to one, so on a truncated domain the highest reachable
        fraction is :meth:`total`), then extracts the contour polylines at
        that level. Requires ``contourpy`` (installed with matplotlib).

        Args:
            rs: Source-area fraction(s) in ``(0, 0.9]`` — a scalar or a
                sequence; values above 1 are read as percentages (``80`` ->
                ``0.8``).

        Returns:
            One dict per fraction, ascending: ``{"r", "level", "fraction",
            "vertices", "closed"}``. ``vertices`` is a list of ``(N, 2)``
            x/y polyline arrays (a level may have several), ``fraction`` is
            the integral actually enclosed at ``level``, and ``closed`` is
            False when any polyline touches the domain edge (the isopleth is
            incomplete — enlarge the domain). A fraction above :meth:`total`
            is unreachable: it warns and yields ``level=nan`` with no
            vertices.
        """
        if isinstance(rs, (int, float)):
            rs = [rs]
        rs = [float(r) / 100.0 if r > 1 else float(r) for r in rs]
        if any(not 0.0 < r <= 0.9 for r in rs):
            raise ValueError(
                f"Source-area fractions must be in (0, 0.9], got {rs}; the "
                "parameterisations are not defined beyond the 90% contour.")
        rs = sorted(rs)

        cell = self.dx * self.dy
        if cell == 0:
            raise ValueError("Cannot contour a degenerate (zero cell area) grid.")
        sf = np.sort(self.f, axis=None)[::-1]
        sf = sf[np.isfinite(sf)]
        csf = np.cumsum(sf) * cell
        captured = float(csf[-1]) if csf.size else 0.0

        out = []
        for r in rs:
            if captured <= 0 or r > captured:
                warnings.warn(
                    f"The {r:.0%} source area is unreachable: the domain "
                    f"captures only {captured:.0%} of the flux. Enlarge "
                    "`domain` to close this contour.", UserWarning,
                    stacklevel=2)
                out.append({"r": r, "level": float("nan"), "fraction": 0.0,
                            "vertices": [], "closed": False})
                continue
            idx = int(np.argmin(np.abs(csf - r)))
            level = float(sf[idx])
            vertices = self._contour_vertices(level)
            # contourpy duplicates the closing vertex of a closed loop
            # exactly, so test exact equality: a relative tolerance would
            # scale with the coordinate origin and mark open, edge-clipped
            # contours "closed" on georeferenced (~1e6 m) grids.
            closed = bool(vertices) and all(
                len(seg) > 2 and np.array_equal(seg[0], seg[-1])
                for seg in vertices)
            out.append({"r": r, "level": level, "fraction": float(csf[idx]),
                        "vertices": vertices, "closed": closed})
        return out

    def _contour_vertices(self, level: float) -> list[np.ndarray]:
        """Polylines of the field at ``level`` as ``(N, 2)`` x/y arrays."""
        try:
            from contourpy import LineType, contour_generator
        except ImportError as exc:  # pragma: no cover - env dependent
            raise ImportError(
                "'contourpy' is required for contour extraction; it is "
                "installed together with matplotlib.") from exc
        gen = contour_generator(x=self.x, y=self.y, z=self.f,
                                line_type=LineType.Separate)
        return [np.asarray(seg, dtype=float) for seg in gen.lines(float(level))]

    def plot(self, ax: Any = None, *, rs: Any = None, cmap: str = "viridis",
             colorbar: bool | None = None, **kwargs: Any) -> Any:
        """Plot the footprint field, optionally with source-area isopleths.

        Requires ``matplotlib``. Never calls ``plt.show()`` — display (or
        save) the returned axes' figure yourself.

        Args:
            ax: Existing matplotlib axes to draw into (a new figure otherwise).
            rs: Source-area fraction(s) to overlay as contour lines
                (see :meth:`contours`).
            cmap: Colormap name for the field.
            colorbar: Attach a colorbar. Default (``None``): only when a new
                figure is created; pass ``True``/``False`` to force either
                way on any axes.
            **kwargs: Forwarded to ``pcolormesh``.

        Returns:
            The matplotlib axes containing the plot.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:  # pragma: no cover - env dependent
            raise ImportError(
                "'matplotlib' is required for Footprint.plot().") from exc
        created_figure = ax is None
        if created_figure:
            _, ax = plt.subplots()
        mesh = ax.pcolormesh(self.x, self.y, self.f, shading="auto",
                             cmap=cmap, **kwargs)
        if colorbar or (colorbar is None and created_figure):
            ax.figure.colorbar(mesh, ax=ax, label="footprint [m$^{-2}$]")
        if rs is not None:
            for contour in self.contours(rs):
                for seg in contour["vertices"]:
                    ax.plot(seg[:, 0], seg[:, 1], color="white",
                            linewidth=1.0)
        tower = self._tower_in_grid_crs()
        if tower is not None:
            ax.plot(*tower, marker="^", color="black", markersize=6,
                    linestyle="none")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_aspect("equal")
        return ax

    def _tower_in_grid_crs(self) -> tuple[float, float] | None:
        """Tower position expressed in the grid's frame, or None if unknown.

        ``self.tower`` is stored in ``tower_crs``, which on a georeferenced
        footprint generally differs from the grid's ``crs`` — plotting it raw
        would place the marker in the wrong spot (or off the map entirely).
        """
        if not self.is_georeferenced:
            return (0.0, 0.0)  # the local frame is tower-centred
        if self.tower is None:
            return None
        if self.tower_crs is None or str(self.tower_crs) == str(self.crs):
            return (float(self.tower[0]), float(self.tower[1]))
        try:
            from pyproj import Transformer
            transformer = Transformer.from_crs(self.tower_crs, self.crs,
                                               always_xy=True)
            tx, ty = transformer.transform(self.tower[0], self.tower[1])
            return (float(tx), float(ty))
        except Exception:  # pyproj missing or CRS unparseable
            logger.debug("Could not transform tower %r from %r to %r.",
                         self.tower, self.tower_crs, self.crs)
            return None

    # -- construction ------------------------------------------------------ #
    @classmethod
    def from_grid(cls, f: np.ndarray, dx: float, dy: float | None = None,
                  **kwargs: Any) -> "Footprint":
        """Build a local footprint from a 2-D field and grid spacing.

        The grid is centred on the tower origin. ``crs`` defaults to ``None``
        (the local frame); pass a projected ``crs`` only if ``f`` is already on
        real coordinates with matching ``x``/``y`` built elsewhere.

        Args:
            f: Footprint field, shape ``(ny, nx)``.
            dx: West-east spacing in metres.
            dy: South-north spacing in metres; defaults to ``dx``.
            **kwargs: Forwarded to :class:`Footprint`.
        """
        f = np.asarray(f, dtype=float)
        if f.ndim != 2:
            raise ValueError(f"f must be 2-D (ny, nx); got shape {f.shape}.")
        dy = dx if dy is None else dy
        ny, nx = f.shape
        x = (np.arange(nx) - (nx - 1) / 2) * dx
        y = (np.arange(ny) - (ny - 1) / 2) * dy
        return cls(f=f, x=x, y=y, **kwargs)

    # -- coordinate frames (optional: pyproj) ------------------------------ #
    def georeference(self, target_crs: str | None = None) -> "Footprint":
        """Place the tower-centred grid into a projected CRS (local -> real).

        The local grid is translated so the tower sits at its real projected
        position; local axes are assumed aligned with the target axes, standard
        for the sub-kilometre domains footprints cover. Requires ``tower`` and
        ``tower_crs``.

        Args:
            target_crs: Projected CRS for the output. When omitted, a
                tower-centred Lambert Azimuthal Equal-Area CRS (:func:`laea_crs`)
                is built automatically, so the grid keeps its tower-centred
                metres and the CRS is recorded on export.

        Returns:
            A georeferenced :class:`Footprint` with real ``x``/``y`` and ``crs``.

        Raises:
            ValueError: If already georeferenced, if ``tower``/``tower_crs`` are
                missing, or if ``time`` is a relative label.
        """
        if self.is_georeferenced:
            raise ValueError("Footprint is already georeferenced.")
        if self.tower is None or self.tower_crs is None:
            raise ValueError(
                "tower and tower_crs are required to georeference a footprint.")
        if _is_relative_label(self.time):
            raise ValueError(
                "cannot georeference a footprint with a relative time label.")

        pyproj = _require("pyproj", "crs")
        if target_crs is None:
            target_crs = _tower_laea(pyproj, self.tower, self.tower_crs)
        transformer = pyproj.Transformer.from_crs(
            self.tower_crs, target_crs, always_xy=True)
        tx, ty = transformer.transform(self.tower[0], self.tower[1])
        return self._replace(x=self.x + tx, y=self.y + ty, crs=target_crs)

    def to_lonlat(self) -> tuple[np.ndarray, np.ndarray]:
        """Return display-only ``(lon, lat)`` grids (EPSG:4326). Requires ``pyproj``.

        The footprint must be georeferenced. Output is curvilinear (2-D) and is
        meant for plotting/labelling, not as a computation grid.
        """
        if not self.is_georeferenced:
            raise ValueError("Footprint must be georeferenced to compute lon/lat.")
        pyproj = _require("pyproj", "crs")
        transformer = pyproj.Transformer.from_crs(self.crs, "EPSG:4326",
                                                  always_xy=True)
        gx, gy = self.meshgrid()
        lon, lat = transformer.transform(gx, gy)
        return np.asarray(lon), np.asarray(lat)

    # -- NetCDF (optional: xarray / netcdf4) ------------------------------- #
    def to_xarray(self) -> "xr.Dataset":
        """Convert to a CF-style :class:`xarray.Dataset`. Requires ``xarray``.

        Georeferenced footprints carry a CF ``grid_mapping`` variable
        (``spatial_ref``) so GDAL/QGIS/rioxarray recognize the CRS.
        """
        xr = _require("xarray", "netcdf")
        ds = xr.Dataset(
            {"footprint": (("y", "x"), self.f,
                           {"units": "m-2", "long_name": "flux footprint"})},
            coords={
                "x": ("x", self.x, _coord_attrs("x", self.is_georeferenced)),
                "y": ("y", self.y, _coord_attrs("y", self.is_georeferenced)),
            },
        )
        ds = ds.assign_coords(**_time_coord(self.time))
        if self.is_georeferenced:
            # As a *coordinate* (not a data variable) so it stays attached
            # when the footprint DataArray is selected - rioxarray/GDAL look
            # it up there.
            ds = ds.assign_coords(
                spatial_ref=_grid_mapping_var(xr, self.crs))
            ds["footprint"].attrs["grid_mapping"] = "spatial_ref"
        ds.attrs.update(self._serializable_attrs())
        ds.attrs.setdefault("Conventions", "CF-1.8")
        return ds

    @classmethod
    def from_xarray(cls, ds: "xr.Dataset") -> "Footprint":
        """Build a footprint from a Dataset produced by :meth:`to_xarray`."""
        a = dict(ds.attrs)
        time = _read_time(ds["time"]) if "time" in ds.coords else None
        return cls(
            f=ds["footprint"].to_numpy(),
            x=ds["x"].to_numpy(), y=ds["y"].to_numpy(),
            time=time, **_read_frame_attrs(a),
            attrs={k: v for k, v in a.items() if k not in _RESERVED_ATTRS})

    def to_netcdf(self, path: str, *, decimals: int | None = None,
                  dtype: str = "int32", **kwargs: Any) -> None:
        """Write to a NetCDF file. Requires ``xarray`` (and a NetCDF engine).

        Args:
            decimals: If set, pack the field as integers scaled by
                ``10**decimals`` (CF ``scale_factor``), shrinking the file;
                read back transparently. Pick a value large enough to preserve
                the small footprint density values (the field is in m**-2).
            dtype: Integer dtype for packing (e.g. ``"int16"`` for more
                compression when the scaled range fits).
        """
        ds = self.to_xarray()
        if decimals is not None:
            _pack_footprint(ds, decimals, dtype)
        ds.to_netcdf(path, **kwargs)

    @classmethod
    def from_netcdf(cls, path: str, **kwargs: Any) -> "Footprint":
        """Read from a NetCDF file. Requires ``xarray`` (and a NetCDF engine)."""
        xr = _require("xarray", "netcdf")
        with xr.open_dataset(path, **kwargs) as ds:
            return cls.from_xarray(ds)

    # -- TIFF (optional: rasterio) ----------------------------------------- #
    def to_tiff(self, path: str, **kwargs: Any) -> None:
        """Write a georeferenced GeoTIFF (north-up). Requires ``rasterio``.

        The footprint must be georeferenced first via :meth:`georeference`.
        """
        rasterio = _require("rasterio", "tiff")
        from rasterio.transform import Affine

        if not self.is_georeferenced:
            raise ValueError(
                "Footprint must be georeferenced before writing a TIFF; "
                "call .georeference(target_crs) first.")

        west = self.x.min() - self.dx / 2
        north = self.y.max() + self.dy / 2
        transform = Affine.translation(west, north) * Affine.scale(self.dx, -self.dy)
        profile = {
            "driver": "GTiff", "dtype": "float32", "count": 1,
            "height": self.ny, "width": self.nx,
            "crs": self.crs, "transform": transform,
            "nodata": np.nan, "compress": "lzw",
        }
        profile.update(kwargs)
        with rasterio.open(path, "w", **profile) as dst:  # row 0 is south -> flip
            dst.write(self.f[::-1].astype("float32"), 1)

    @classmethod
    def from_tiff(cls, path: str, **kwargs: Any) -> "Footprint":
        """Read a georeferenced footprint from a GeoTIFF. Requires ``rasterio``."""
        rasterio = _require("rasterio", "tiff")
        with rasterio.open(path, **kwargs) as src:
            band = np.asarray(src.read(1), dtype=float)[::-1]  # back to south-first
            t = src.transform
            dx, dy = float(t.a), float(-t.e)
            west, north = float(t.c), float(t.f)
            x = west + dx / 2 + np.arange(src.width) * dx
            y = north - dy / 2 - np.arange(src.height) * dy
            crs = str(src.crs) if src.crs is not None else None
        return cls(f=band, x=x, y=np.sort(y), crs=crs)

    # -- Shapefile (optional: fiona) --------------------------------------- #
    def to_shapefile(self, path: str, rs: Any = (0.5, 0.8),
                     **kwargs: Any) -> None:
        """Write source-area isopleths to an ESRI Shapefile. Requires ``fiona``.

        Each closed contour from :meth:`contours` becomes one polygon record
        with properties ``r`` (source-area fraction) and ``level`` (field
        value). Contours that touch the domain edge are skipped with a
        warning — enlarge the model domain to close them. When the footprint
        is georeferenced the CRS is written alongside.

        Args:
            path: Destination ``.shp`` path.
            rs: Source-area fraction(s) to export (see :meth:`contours`).
            **kwargs: Forwarded to ``fiona.open``.
        """
        fiona = _require("fiona", "shapefile")
        open_kwargs: dict[str, Any] = {
            "driver": "ESRI Shapefile",
            "schema": {"geometry": "Polygon",
                       "properties": {"r": "float", "level": "float"}},
        }
        if self.crs is not None:
            try:
                from pyproj import CRS
                open_kwargs["crs_wkt"] = CRS.from_user_input(self.crs).to_wkt()
            except Exception:  # pyproj missing or crs unparseable
                logger.debug("Could not expand crs %r for the shapefile.",
                             self.crs)
        open_kwargs.update(kwargs)

        contours = self.contours(rs)
        with fiona.open(str(path), "w", **open_kwargs) as dst:
            for contour in sorted(contours, key=lambda c: c["r"],
                                  reverse=True):
                if not contour["closed"]:
                    if contour["vertices"]:  # unreachable ones warned already
                        warnings.warn(
                            f"Skipping the open {contour['r']:.0%} contour "
                            "(it touches the domain edge).", UserWarning,
                            stacklevel=2)
                    continue
                for rings in _nest_rings(contour["vertices"]):
                    dst.write({
                        "geometry": {"type": "Polygon",
                                     "coordinates": rings},
                        "properties": {"r": float(contour["r"]),
                                       "level": float(contour["level"])},
                    })

    # -- internals --------------------------------------------------------- #
    @staticmethod
    def _coerce_attr(value: Any) -> Any:
        """Coerce an attrs value to a NetCDF-safe type (str/number/array).

        Timestamps become ISO strings and anything exotic falls back to
        ``str``; without this, e.g. a ``pd.Timestamp`` group label makes
        ``to_netcdf`` fail on the documented by-timestamp batch workflow.
        """
        if isinstance(value, datetime):  # covers pd.Timestamp (a subclass)
            return value.isoformat()
        if isinstance(value, np.datetime64):
            return str(value)
        # netCDF attrs have no boolean type; np.bool_ is an np.generic that
        # xarray rejects, so it must be caught before the pass-through branch.
        if isinstance(value, (bool, np.bool_)):
            return int(value)
        if isinstance(value, (str, bytes, int, float, np.ndarray, np.generic)):
            return value
        if isinstance(value, (list, tuple)) and all(
                isinstance(v, (str, int, float, np.generic)) for v in value):
            return [Footprint._coerce_attr(v) for v in value]
        return str(value)

    def _serializable_attrs(self) -> dict[str, Any]:
        out = {k: self._coerce_attr(v)
               for k, v in self.attrs.items() if v is not None}
        if self.crs is not None:
            out.setdefault("crs", self.crs)
            try:  # enrich with wkt/proj4 when pyproj is available
                from pyproj import CRS
                _crs = CRS.from_user_input(self.crs)
                out.setdefault("crs_wkt", _crs.to_wkt())
                out.setdefault("crs_proj4", _crs.to_proj4())
            except Exception:  # pyproj missing or crs unparseable -> keep the string
                logger.debug("Could not expand crs %r to wkt/proj4.", self.crs)
        if self.tower is not None:
            out.setdefault("tower_x", self.tower[0])
            out.setdefault("tower_y", self.tower[1])
        if self.tower_crs is not None:
            out.setdefault("tower_crs", self.tower_crs)
        if self.n is not None:
            out.setdefault("n_records", self.n)
        return out

    def _replace(self, **changes: Any) -> "Footprint":
        """Return a copy with the given attributes replaced."""
        current = {
            "f": self.f, "x": self.x, "y": self.y, "time": self.time,
            "crs": self.crs, "tower": self.tower, "tower_crs": self.tower_crs,
            "n": self.n, "attrs": dict(self.attrs),
        }
        current.update(changes)
        return Footprint(**current)


# --------------------------------------------------------------------------- #
# Shared NetCDF attribute helpers                                             #
# --------------------------------------------------------------------------- #
_RESERVED_ATTRS = {"crs", "tower_x", "tower_y", "tower_crs", "n_records",
                   "Conventions"}


def _pack_footprint(ds: "xr.Dataset", decimals: int, dtype: str = "int32") -> "xr.Dataset":
    """Encode the footprint variable as scaled integers (CF ``scale_factor``).

    Stores ``round(f * 10**decimals)`` as ``dtype`` with ``scale_factor =
    10**-decimals``, so the file is far smaller than float64 and any CF reader
    (including :meth:`Footprint.from_netcdf`) unpacks it transparently. Choose
    ``decimals`` large enough to preserve the small footprint density values.
    """
    info = np.iinfo(np.dtype(dtype))
    ds["footprint"].encoding.update({
        "dtype": dtype,
        "scale_factor": 10.0 ** (-decimals),
        "_FillValue": info.min,
    })
    return ds


def _coord_attrs(axis: str, georeferenced: bool) -> dict[str, str]:
    attrs = {"units": "m", "axis": axis.upper(),
             "long_name": f"{axis} ({'projected' if georeferenced else 'from tower'})"}
    if georeferenced:
        # The CF projection_* standard names are only true of a projected CRS;
        # local tower-relative metres must not claim them.
        attrs["standard_name"] = {
            "x": "projection_x_coordinate", "y": "projection_y_coordinate",
        }[axis]
    return attrs


def _grid_mapping_var(xr_mod: Any, crs: str) -> "xr.DataArray":
    """Scalar CF grid-mapping variable carrying the CRS (rioxarray-style)."""
    try:
        from pyproj import CRS
        _crs = CRS.from_user_input(crs)
        wkt = _crs.to_wkt()
        attrs = {"crs_wkt": wkt, "spatial_ref": wkt}
        try:
            attrs.update({k: v for k, v in _crs.to_cf().items()
                          if v is not None})
        except Exception:  # to_cf can fail for exotic CRSes; wkt suffices
            pass
    except Exception:  # pyproj missing or crs unparseable -> keep the string
        attrs = {"crs_wkt": str(crs)}
    return xr_mod.DataArray(0, attrs=attrs)


def _time_coord(time: datetime | float | None) -> dict[str, Any]:
    if time is None:
        return {}
    if isinstance(time, datetime):
        return {"time": np.datetime64(time)}
    import xarray as xr  # only reached from to_xarray, so xarray is present
    return {"time": xr.DataArray(
        time, attrs={"units": "1",
                     "long_name": "relative time label (arbitrary origin)"})}


def _read_time(coord: "xr.DataArray") -> datetime | float:
    values = np.asarray(coord.values)
    if np.issubdtype(values.dtype, np.datetime64):
        return values.astype("datetime64[s]").astype(datetime)
    return float(values)


def _read_frame_attrs(a: dict[str, Any]) -> dict[str, Any]:
    tower = None
    if "tower_x" in a and "tower_y" in a:
        tower = (float(a["tower_x"]), float(a["tower_y"]))
    return {
        "crs": a.get("crs"),
        "tower": tower,
        "tower_crs": a.get("tower_crs"),
        "n": a.get("n_records"),
    }


# --------------------------------------------------------------------------- #
# FootprintSeries                                                             #
# --------------------------------------------------------------------------- #
class FootprintSeries:
    """A time-ordered stack of footprints ``(time, y, x)`` on one shared grid.

    All members must share the same grid (``x``/``y``) and frame (``crs``).
    Aggregating collapses the stack to a 2-D climatology.
    """

    __slots__ = ("footprints",)

    def __init__(self, footprints: list[Footprint]):
        """Validate that all footprints share one grid and frame.

        Args:
            footprints: Footprints sharing identical ``x``/``y`` and ``crs``.

        Raises:
            ValueError: If empty or the grids/frames are inconsistent.
        """
        if not footprints:
            raise ValueError("FootprintSeries requires at least one footprint.")
        ref = footprints[0]
        for fp in footprints[1:]:
            if fp.f.shape != ref.f.shape or not np.allclose(fp.x, ref.x) \
                    or not np.allclose(fp.y, ref.y):
                raise ValueError("All footprints must share the same grid.")
            if fp.crs != ref.crs:
                raise ValueError("All footprints must share the same crs.")
        self.footprints = list(footprints)

    def __len__(self) -> int:
        return len(self.footprints)

    def __iter__(self) -> Iterator[Footprint]:
        return iter(self.footprints)

    def __getitem__(self, i: int) -> Footprint:
        return self.footprints[i]

    @property
    def nt(self) -> int:
        """Number of time steps."""
        return len(self.footprints)

    @property
    def crs(self) -> str | None:
        """Shared CRS (``None`` if local)."""
        return self.footprints[0].crs

    @property
    def is_georeferenced(self) -> bool:
        """True if the shared grid is in a projected CRS."""
        return self.footprints[0].is_georeferenced

    @property
    def x(self) -> np.ndarray:
        """Shared west-east coordinate vector."""
        return self.footprints[0].x

    @property
    def y(self) -> np.ndarray:
        """Shared south-north coordinate vector."""
        return self.footprints[0].y

    @property
    def times(self) -> list[datetime | float | None]:
        """Per-step time labels, in order."""
        return [fp.time for fp in self.footprints]

    def stack(self) -> np.ndarray:
        """Return the ``(nt, ny, nx)`` array of fields."""
        return np.stack([fp.f for fp in self.footprints])

    def aggregate(self, *, smooth: bool = True) -> Footprint:
        """Collapse the stack to a 2-D climatology (mean over time).

        Args:
            smooth: If True, apply the standard 3x3 kernel twice (needs scipy).

        Returns:
            A climatological :class:`Footprint` (``time=None``) on the shared
            grid, with ``n`` summed across members when available.
        """
        fclim = np.nanmean(self.stack(), axis=0)
        if smooth:
            fclim = smooth_field(fclim)
        counts = [fp.n for fp in self.footprints if fp.n is not None]
        ref = self.footprints[0]
        # Carry provenance the members agree on (e.g. 'model',
        # 'estimated_inputs') onto the climatology; per-member values (e.g.
        # 'group', 'captured_fraction') are dropped rather than fabricated.
        # 'history' is never equality-carried (member timestamps differ by
        # wall clock); the climatology gets its own fresh entry instead.
        attrs = {}
        for key, value in ref.attrs.items():
            if key == "history":
                continue
            try:
                if all(bool(np.all(fp.attrs.get(key) == value))
                       for fp in self.footprints):
                    attrs[key] = value
            except (TypeError, ValueError):
                continue
        from .version import __version__
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        attrs["history"] = (f"{stamp} aggregated {len(self.footprints)} "
                            f"footprint(s) by fluxprint {__version__}")
        return Footprint(
            f=fclim, x=ref.x.copy(), y=ref.y.copy(), time=None, crs=ref.crs,
            tower=ref.tower, tower_crs=ref.tower_crs,
            n=int(sum(counts)) if counts else None, attrs=attrs)

    def georeference(self, target_crs: str | None = None) -> "FootprintSeries":
        """Georeference every member onto the same projected grid.

        Computes the translation once from the shared tower position and applies
        it to all members. ``target_crs`` defaults to a tower-centred LAEA
        (:func:`laea_crs`). Requires ``datetime`` timestamps on every member.
        """
        if self.is_georeferenced:
            raise ValueError("FootprintSeries is already georeferenced.")
        if any(not isinstance(fp.time, datetime) for fp in self.footprints):
            raise ValueError(
                "every footprint needs a datetime timestamp to georeference "
                "a series.")
        ref = self.footprints[0]
        if ref.tower is None or ref.tower_crs is None:
            raise ValueError("tower and tower_crs are required to georeference.")

        pyproj = _require("pyproj", "crs")
        if target_crs is None:
            target_crs = _tower_laea(pyproj, ref.tower, ref.tower_crs)
        transformer = pyproj.Transformer.from_crs(
            ref.tower_crs, target_crs, always_xy=True)
        tx, ty = transformer.transform(ref.tower[0], ref.tower[1])
        new_x, new_y = ref.x + tx, ref.y + ty
        return FootprintSeries([
            fp._replace(x=new_x, y=new_y, crs=target_crs)
            for fp in self.footprints])

    # -- NetCDF (optional: xarray / netcdf4) ------------------------------- #
    def to_xarray(self) -> "xr.Dataset":
        """Convert to a ``(time, y, x)`` :class:`xarray.Dataset`. Requires ``xarray``.

        Only grid-level metadata survives: global attrs are taken from the
        first member, so attrs that differ per member (e.g. ``group``,
        ``captured_fraction``) are not round-tripped. Per-step ``n`` is kept
        as the ``n_records`` variable.
        """
        xr = _require("xarray", "netcdf")
        ref = self.footprints[0]
        data = {"footprint": (("time", "y", "x"), self.stack(),
                              {"units": "m-2", "long_name": "flux footprint"})}
        counts = [fp.n for fp in self.footprints]
        if all(c is not None for c in counts):
            data["n_records"] = (("time",), np.asarray(counts, dtype="int64"))
        ds = xr.Dataset(
            data,
            coords={
                "time": ("time", _time_axis(self.times)),
                "x": ("x", ref.x, _coord_attrs("x", self.is_georeferenced)),
                "y": ("y", ref.y, _coord_attrs("y", self.is_georeferenced)),
            },
        )
        if self.is_georeferenced:
            ds = ds.assign_coords(
                spatial_ref=_grid_mapping_var(xr, ref.crs))
            ds["footprint"].attrs["grid_mapping"] = "spatial_ref"
        ds.attrs.update({k: v for k, v in ref._serializable_attrs().items()
                         if k != "n_records"})
        ds.attrs.setdefault("Conventions", "CF-1.8")
        return ds

    @classmethod
    def from_xarray(cls, ds: "xr.Dataset") -> "FootprintSeries":
        """Build a series from a Dataset produced by :meth:`to_xarray`."""
        a = dict(ds.attrs)
        frame = _read_frame_attrs(a)
        frame.pop("n", None)
        extra = {k: v for k, v in a.items() if k not in _RESERVED_ATTRS}
        counts = ds["n_records"].to_numpy() if "n_records" in ds.variables else None
        times = ds["time"]
        footprints = []
        for i in range(ds.sizes["time"]):
            footprints.append(Footprint(
                f=ds["footprint"].isel(time=i).to_numpy(),
                x=ds["x"].to_numpy(), y=ds["y"].to_numpy(),
                time=_read_time(times.isel(time=i)), **frame,
                n=int(counts[i]) if counts is not None else None,
                attrs=dict(extra)))
        return cls(footprints)

    def to_netcdf(self, path: str, *, decimals: int | None = None,
                  dtype: str = "int32", **kwargs: Any) -> None:
        """Write the series to a NetCDF file. Requires ``xarray`` (+ engine).

        ``decimals``/``dtype`` pack the field as scaled integers exactly as in
        :meth:`Footprint.to_netcdf`.
        """
        ds = self.to_xarray()
        if decimals is not None:
            _pack_footprint(ds, decimals, dtype)
        ds.to_netcdf(path, **kwargs)

    @classmethod
    def from_netcdf(cls, path: str, **kwargs: Any) -> "FootprintSeries":
        """Read a series from a NetCDF file. Requires ``xarray`` (+ engine)."""
        xr = _require("xarray", "netcdf")
        with xr.open_dataset(path, **kwargs) as ds:
            return cls.from_xarray(ds)


def _time_axis(times: list[datetime | float | None]) -> np.ndarray:
    """Build a time coordinate array from per-step labels."""
    if all(isinstance(t, datetime) for t in times):
        return np.array([np.datetime64(t) for t in times])
    if all(_is_relative_label(t) for t in times):
        return np.asarray(times, dtype=float)
    # Mixed/None labels: fall back to an integer index.
    return np.arange(len(times))
