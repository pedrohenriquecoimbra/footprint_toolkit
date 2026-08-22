# Contributing to FluxPrint

Contributions are welcome. Fork the repository, create a branch for your
change, and open a pull request.

## Development setup

```bash
git clone https://github.com/pedrohenriquecoimbra/fluxprint
cd fluxprint
pip install -e .[dev]
python -m pytest tests/ -q
```

The full suite runs in a few seconds. CI runs it on Python 3.10-3.13, plus a
dedicated job asserting that every registered footprint model reproduces its
original (vendored) reference implementation **exactly**.

## Adding a footprint model

A model is a keyword-only callable returning one local-frame `Footprint`
(see `fluxprint/model/base.py` for the protocol, and the Kljun adapter in
`fluxprint/model/Kljun_et_al_2015.py` for the pattern):

1. implement/port the model; keep the reference implementation vendored
   alongside it if you rewrote the code;
2. register it with `@register_model("name", description=..., citation=...,
   doi=...)` and stamp provenance into the returned footprint's `attrs`;
3. add a reference oracle in `tests/test_reference_regression.py`
   (`REFERENCE_ORACLES`) - the suite fails if a registered model has no
   reference check;
4. validate inputs per record (reject NaN/None; see
   `exceptions.check_ffp_inputs`).

## Conventions

- Behavior changes and deprecations go into `CHANGELOG.md` (Keep-a-Changelog).
- The batch path never fabricates inputs silently: crude constant fills are
  opt-in (`fill_all=True`) and warn.
- New public API needs tests; scientific claims need a pinned numeric test.
