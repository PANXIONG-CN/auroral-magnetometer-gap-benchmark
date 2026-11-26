# auroral-magnetometer-gap-benchmark

Reproducibility code for *Where Does the Error Come From When a Missing Auroral-Zone Magnetometer Is Reconstructed? Station Baselines, Network Geometry, and Temporal Information* (submitted to *Space Weather*).

Zenodo archive (frozen locked-test outputs, `MANIFEST.csv`, and `v1.0.0` snapshot): [10.5281/zenodo.24564786](https://doi.org/10.5281/zenodo.24564786)

## Setup

Requires [uv](https://docs.astral.sh/uv/) (Python 3.13 pinned by `.python-version` and `uv.lock`):

```bash
uv sync --frozen
uv run pytest
```

Tests that require locally downloaded magnetometer or field-model files are skipped automatically until raw data are fetched.

## Reproduce Figures and Tables

Download `e011_locked_results_c462853_cfgebe313.tar.gz` from [Zenodo](https://doi.org/10.5281/zenodo.24564786) and extract it at the repository root:

```bash
tar -xzf e011_locked_results_c462853_cfgebe313.tar.gz
uv run scripts/make_figures.py f1 f2 f3 f4 f5
uv run scripts/export_latex_assets.py
```

Outputs are written to `figures/` and `paper/agu_latex/tables/`.

## Raw Data (Optional Full Re-run)

Upstream raw observations are not redistributed here; all 547 files, official URLs, and SHA-256 checksums are recorded in `data/raw/MANIFEST.csv`. To fetch and verify them from FMI IMAGE, GFZ Potsdam, WDC Kyoto, NOAA NCEI, and DTU Space:

```bash
uv run scripts/download_data.py indices
uv run scripts/download_data.py stations
uv run scripts/download_data.py igrf
uv run scripts/download_data.py chaos
uv run scripts/download_data.py image --groups training validation development test
uv run scripts/download_data.py verify
```

Locked test storms (`config/events.yaml`) cannot be read by default; development-storm checks run via `uv run scripts/run_locked_test.py --dev`.

## License

Code is released under the [MIT License](LICENSE). Archived benchmark outputs on Zenodo are released under CC-BY-4.0. Upstream magnetometer and geomagnetic-index data remain subject to their respective provider rules.
