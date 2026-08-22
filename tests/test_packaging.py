"""Packaging-level checks: import weight stays under control.

The optional-extras split requires `import fluxprint` not to touch the heavy
geo/plotting stack; this pins the lazy-import work so an eager import cannot
sneak back in (the same check runs as a CI step).
"""
from __future__ import annotations

import subprocess
import sys

HEAVY = ("rasterio", "fiona", "shapely", "xarray", "pyproj", "matplotlib")


def test_import_fluxprint_does_not_load_geo_stack():
    code = (
        "import fluxprint, sys; "
        f"loaded = [m for m in {HEAVY!r} if m in sys.modules]; "
        "assert not loaded, f'import fluxprint eagerly loaded {loaded}'"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_lazy_submodules_still_reachable():
    code = (
        "import fluxprint; "
        "assert callable(fluxprint.utils.get_contour_levels); "
        "assert hasattr(fluxprint.template, 'DEFAULT_ATTRS')"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
