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

## Cutting a release

1. Bump `fluxprint/version.py`, and `CITATION.cff` (`version` +
   `date-released`); date the release's heading in `CHANGELOG.md`.
2. Commit, then tag and push:

   ```bash
   git tag -a v0.4.0 -m "fluxprint 0.4.0" && git push origin v0.4.0
   ```

The `publish` workflow takes it from there: it refuses to continue if the tag
does not match `version.py`, builds the sdist and wheel, runs the whole suite
**against the built wheel** (plus the model reference-equivalence check), and
uploads to PyPI via Trusted Publishing. Pre-release versions are fine — a
`v0.4.0a1` tag publishes an alpha that plain `pip install fluxprint` will not
pick up. For a dry run without tagging, use Actions → publish → Run workflow →
target `testpypi`.

Trusted Publishing needs a one-time setup on PyPI (project → Publishing → add
a GitHub publisher for `publish.yml`, environment `pypi`); the workflow header
documents the exact fields.

## Conventions

- Behavior changes and deprecations go into `CHANGELOG.md` (Keep-a-Changelog).
- The batch path never fabricates inputs silently: crude constant fills are
  opt-in (`fill_all=True`) and warn.
- New public API needs tests; scientific claims need a pinned numeric test.
