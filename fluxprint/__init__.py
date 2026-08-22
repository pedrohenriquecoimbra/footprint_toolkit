"""FluxPrint: flux footprint models for eddy covariance data analysis."""
import importlib
import logging

from . import core, io, commons, model
from .core import *  # noqa: F401,F403 (public API, bounded by core.__all__)
from .io import *    # noqa: F401,F403 (public API, bounded by io.__all__)
from .footprint import Footprint, FootprintSeries
from .version import __version__

# Libraries should not configure logging; attach a no-op handler so the package
# emits nothing unless the application configures the "fluxprint" logger.
logging.getLogger("fluxprint").addHandler(logging.NullHandler())

#: Submodules loaded on first attribute access (PEP 562): `utils` pulls the
#: heavy geo/plotting stack (rasterio, xarray, pyproj, matplotlib) at import,
#: so it must not load with `import fluxprint`.
_LAZY_SUBMODULES = ("utils", "template")


def __getattr__(name):
    if name in _LAZY_SUBMODULES:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    *core.__all__,
    *io.__all__,
    "Footprint",
    "FootprintSeries",
    "model",
    "__version__",
]
