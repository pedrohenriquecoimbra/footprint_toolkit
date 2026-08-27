"""Model-agnostic output-grid machinery for footprint models.

Every footprint model computes on the same kind of fixed, regular,
tower-centred grid; this module owns that grid so models own only physics.
:func:`resolve_grid` reproduces the FFP reference's ``domain``/``dx``/``dy``/
``nx``/``ny`` reconciliation rules **verbatim** (the registered Kljun port is
pinned bitwise to the vendored reference, so the resolution quirks — ``int()``
truncation, the nx-only branch deriving ``dy`` from ``nx``, a non-list
``domain`` being ignored — are kept deliberately). :class:`GridSpec` describes
the resolved grid (shape known without any compute), and :class:`GridContext`
caches the concrete coordinate arrays for one model call.

**Wind-direction convention** (shared by all models): ``wind_dir`` is the
direction the wind comes *from*, in degrees clockwise from north. The output
grid never rotates — a model evaluates its wind-aligned footprint at the
wind-frame coordinates of the fixed grid. The along-wind coordinate is
positive toward the wind source (the fetch). Two equivalent formulations:

* polar (the Kljun kernel): ``rho * cos(theta - w)`` along-wind and
  ``rho * sin(theta - w)`` crosswind, with ``theta = arctan2(x, y)`` (azimuth
  clockwise from north) and ``w = wind_dir * pi / 180`` — see
  :func:`rotate_theta`;
* cartesian (:func:`to_wind_frame`): ``x*sin(w) + y*cos(w)`` along-wind and
  ``x*cos(w) - y*sin(w)`` crosswind.

They are equal in exact arithmetic but **not bitwise equal** in float64;
the Kljun kernel keeps the polar form (its output is pinned to the
reference), while new models should use :func:`to_wind_frame`.

Depends only on :mod:`numpy`.
"""
from __future__ import annotations

import numbers
from dataclasses import dataclass

import numpy as np

__all__ = ["GridSpec", "GridContext", "resolve_grid", "rotate_theta",
           "to_wind_frame"]


@dataclass(frozen=True)
class GridSpec:
    """A resolved output grid: domain bounds, spacing, and element counts.

    ``nx``/``ny`` follow the FFP convention of counting grid *elements*
    (cells), so the coordinate axes have ``nx + 1`` / ``ny + 1`` points
    (fenceposts) and the field has shape :attr:`shape`.
    """

    xmin: float
    xmax: float
    ymin: float
    ymax: float
    dx: float
    dy: float
    nx: int
    ny: int

    @property
    def domain(self) -> list:
        """``[xmin, xmax, ymin, ymax]`` (the FFP ``domain`` argument)."""
        return [self.xmin, self.xmax, self.ymin, self.ymax]

    @property
    def shape(self) -> tuple[int, int]:
        """Output field shape ``(ny + 1, nx + 1)`` — known without compute."""
        return (self.ny + 1, self.nx + 1)

    def axes(self) -> tuple[np.ndarray, np.ndarray]:
        """1-D cell-centre coordinate vectors ``(x, y)``."""
        x = np.linspace(self.xmin, self.xmax, self.nx + 1)
        y = np.linspace(self.ymin, self.ymax, self.ny + 1)
        return x, y

    def meshgrid(self) -> tuple[np.ndarray, np.ndarray]:
        """2-D coordinate grids ``(x_2d, y_2d)`` matching :attr:`shape`."""
        x, y = self.axes()
        return np.meshgrid(x, y)


def resolve_grid(domain=None, dx=None, dy=None, nx=None, ny=None) -> GridSpec:
    """Reconcile ``domain``/``dx``/``dy``/``nx``/``ny`` into a :class:`GridSpec`.

    This is the FFP reference's resolution logic, moved verbatim (quirks
    included) from the Kljun port:

    * nothing passed -> a 2 km tower-centred square at 2 m (a 1001x1001 grid);
    * ``domain`` (a list of 4) takes precedence; with ``dx``/``dy`` the counts
      are ``int()``-truncated, otherwise ``nx``/``ny`` (default 1000) set the
      spacing;
    * without ``domain``: ``dx`` and ``nx`` together size the domain;
      ``dx`` alone (or ``nx`` alone) applies to the default 2 km domain.

    A ``dx`` without ``dy`` (or ``nx`` without ``ny``) is squared up; a
    ``domain`` that is not a list of length 4 is ignored.
    """
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

    return GridSpec(xmin=domain[0], xmax=domain[1], ymin=domain[2],
                    ymax=domain[3], dx=dx, dy=dy, nx=nx, ny=ny)


class GridContext:
    """Concrete coordinate arrays for one model call, computed once.

    Wraps a :class:`GridSpec` with the cartesian axes and meshes eagerly and
    the polar coordinates lazily (only polar-formulated kernels need them).
    Kernels must treat every array as read-only.
    """

    __slots__ = ("spec", "x", "y", "x_2d", "y_2d", "_rho", "_theta")

    def __init__(self, spec: GridSpec):
        self.spec = spec
        self.x, self.y = spec.axes()
        self.x_2d, self.y_2d = np.meshgrid(self.x, self.y)
        self._rho = None
        self._theta = None

    @property
    def rho(self) -> np.ndarray:
        """Distance from the tower, ``sqrt(x_2d**2 + y_2d**2)``."""
        if self._rho is None:
            self._rho = np.sqrt(self.x_2d**2 + self.y_2d**2)
        return self._rho

    @property
    def theta(self) -> np.ndarray:
        """Azimuth ``arctan2(x_2d, y_2d)``: north up, increasing clockwise."""
        if self._theta is None:
            self._theta = np.arctan2(self.x_2d, self.y_2d)
        return self._theta


def rotate_theta(theta: np.ndarray, wind_dir: float) -> np.ndarray:
    """Azimuth relative to the wind direction (polar rotation, radians).

    ``theta - wind_dir * pi / 180`` — the FFP reference's rotation, moved
    verbatim. ``rho * cos(rotate_theta(...))`` is the along-wind coordinate
    (positive toward the wind source), ``rho * sin(...)`` the crosswind one.
    """
    return theta - wind_dir * np.pi / 180.


def to_wind_frame(x_2d: np.ndarray, y_2d: np.ndarray,
                  wind_dir: float) -> tuple[np.ndarray, np.ndarray]:
    """Wind-frame coordinates of the fixed grid (cartesian formulation).

    Returns ``(along_wind, crosswind)`` where ``along_wind = x*sin(w) +
    y*cos(w)`` is positive toward the wind source (the fetch) and
    ``crosswind = x*cos(w) - y*sin(w)``, with ``w = wind_dir`` in radians.
    Mathematically equal to the polar form (see :func:`rotate_theta`) but
    not bitwise equal in float64 — new models should use this; the Kljun
    kernel keeps the polar form its reference pin requires.
    """
    w = wind_dir * np.pi / 180.
    along = x_2d * np.sin(w) + y_2d * np.cos(w)
    cross = x_2d * np.cos(w) - y_2d * np.sin(w)
    return along, cross
