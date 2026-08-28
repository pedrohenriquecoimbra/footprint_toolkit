"""Regenerate the golden snapshot behind the ``hsieh2000`` reference oracle.

The snapshot (``hsieh2000_reference.npz``) pins the registered Hsieh model's
output in ``tests/test_reference_regression.py``. There is no vendored
original for this model, so the pin is a regression baseline, not an
independent derivation — the tie to the published equations lives in
``tests/test_hsieh_reference.py``.

Run this ONLY after an *intended* change to the Hsieh physics, and record the
change in the changelog::

    python tests/data/regenerate_hsieh_reference.py
"""
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2]))  # repo root -> import fluxprint
sys.path.insert(0, str(_HERE.parents[1]))  # tests dir -> import the battery

import test_reference_regression as trr  # noqa: E402

from fluxprint.model import get_model  # noqa: E402


def main() -> None:
    model = get_model("hsieh2000")
    unsupported = trr.UNSUPPORTED_CASES.get("hsieh2000", set())
    arrays = {}
    for case, met in trr.CASES.items():
        if case in unsupported:
            continue
        fp = model(**met, **trr.GRID)
        arrays[f"{case}_f"] = fp.f
        arrays[f"{case}_n"] = np.int64(fp.n)
    out = _HERE.with_name("hsieh2000_reference.npz")
    np.savez_compressed(out, **arrays)
    print(f"wrote {out} ({out.stat().st_size} bytes, "
          f"{len(arrays) // 2} case(s))")


if __name__ == "__main__":
    main()
