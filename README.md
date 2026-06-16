# TDA Finance Mapper

[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue.svg)](https://jaimecrz3.github.io/tda-finance-mapper/)
[![PyPI - Version](https://img.shields.io/pypi/v/tda-finance-mapper?color=blue)](https://pypi.org/project/tda-finance-mapper/)

`tda-finance-mapper` is a Python package developed as part of an academic project on the application of Topological Data Analysis (TDA) to financial data.

The package provides tools to build Mapper-based portfolio strategies, compute persistent-homology regime signals and evaluate the resulting portfolios through causal backtesting.

## Overview

The project studies whether topological information extracted from financial return windows can be used for:

1. market-structure analysis;
2. regime detection;
3. portfolio construction;
4. comparison against a simple equal-weight benchmark.

The main implemented models are:

- **Mapper portfolio**: assets are represented by recent return vectors, a Mapper graph is built, and the graph structure is transformed into portfolio weights.
- **Mapper + persistent homology**: Mapper remains the main portfolio construction method, while persistent homology acts as a regime-control signal.
- **Equal-weight benchmark**: all available assets receive the same weight and are evaluated with the same backtesting protocol.

## Disclaimer

This project is for academic and research purposes only. It is not financial advice, investment advice or a production trading system. The results are intended to illustrate and evaluate a methodological pipeline, not to recommend real investment decisions.

## Installation

There are two ways to install the package.

### Install from PyPI

This is the recommended option if you only want to use the package.

```bash
python -m venv .venv
source .venv/bin/activate
pip install tda-finance-mapper
````

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install tda-finance-mapper
```

### Install from the repository

This option is recommended if you want to reproduce the experiments, inspect the source code, modify the package or build the documentation.

```bash
git clone https://github.com/jaimecrz3/tda-finance-mapper.git
cd tda-finance-mapper
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows:

```bash
git clone https://github.com/jaimecrz3/tda-finance-mapper.git
cd tda-finance-mapper
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

The `requirements.txt` file installs the project in editable mode with:

```txt
-e .
```

This means that changes made in the source code are immediately available without reinstalling the package.

For development tools such as `flake8`, `pytest`, `sphinx`, `build` and `twine`, install the optional development dependencies:

```bash
pip install -e .[dev]
```


## Project structure

```text
tda-finance-mapper/
├── data/
├── docs/
├── results_49_Industry_Portfolios/
├── results_SP500_CRSP/
├── scripts/
├── src/
│   └── tda_finance/
│       ├── data_preprocessing/
│       ├── experiments/
│       ├── portfolio/
│       └── tda/
├── pyproject.toml
├── requirements.txt
├── LICENSE
└── README.md
```

The main package is located under `src/tda_finance`.

The `scripts/` folder contains auxiliary scripts used during data preparation. These scripts are not part of the main package API.

## Main modules

### `tda_finance.tda.mapper_clustering`

Builds Mapper graphs from financial price windows and converts Mapper clusters into portfolio weights.

### `tda_finance.tda.persistence_diagrams`

Computes correlation-based distance matrices and persistent-homology diagrams.

### `tda_finance.tda.persistence_features`

Extracts summary features from persistence diagrams.

### `tda_finance.tda.regime_detection`

Computes persistence-landscape norms and detects topological anomalies.

### `tda_finance.portfolio.backtest_engine`

Runs causal long-only backtests, computes portfolio returns, turnover and performance metrics.

### `tda_finance.data_preprocessing`

Contains utilities to preprocess Kenneth French 49 Industry Portfolios and S&P 500 CRSP monthly data.

## Minimal API example

The following example shows the basic use of the package API with a generic price matrix.

```python
import pandas as pd

from tda_finance.portfolio.backtest_engine import backtest_tda, perf_summary
from tda_finance.tda.mapper_clustering import MapperParams

prices = pd.read_csv(
    "data/prices.csv",
    index_col=0,
    parse_dates=True,
)

params = MapperParams(
    pca_var=0.80,
    umap_dim=1,
    n_cubes=12,
    perc_overlap=0.25,
    clusterer="haca",
    haca_distance_threshold=0.6,
    haca_linkage="average",
    random_state=1,
)

result = backtest_tda(
    prices=prices,
    lookback_days=60,
    rebalance_days=3,
    params=params,
    tc_bps=5.0,
    use_ph_control=False,
)

metrics = perf_summary(
    result["port_ret"],
    periods_per_year=12,
)

print(metrics)
```

This is only a minimal usage example. The full experimental protocol is implemented in the experiment script described below.

## Documentation

The documentation is built with Sphinx.

Online documentation is available at:

```text
https://jaimecrz3.github.io/tda-finance-mapper/
```

To build the documentation locally:

```bash
pip install -e .[dev]
sphinx-build -b html docs/source docs/_build/html
```

Then open:

```text
docs/_build/html/index.html
```

## Running the final experiments

The final experiments can be run from the project root with:

```bash
python -m tda_finance.experiments.run_mapper_ph_experiments
```

The script compares:

1. Mapper;
2. Mapper with persistent-homology regime control;
3. equal-weight benchmark.

The selected dataset is configured inside the script.

## Data preparation

Auxiliary scripts used to prepare the S&P 500 CRSP data are located in:

```text
scripts/data_preparation/
```

Example:

```bash
python scripts/data_preparation/prepare_prices_sp500.py
python scripts/data_preparation/make_monthly_sp500.py
```

These scripts are included to make the data preparation process more transparent, but they are not part of the main package API.

## Data availability

The repository does not include the raw financial datasets used in the experiments.

The Kenneth French 49 Industry Portfolios and Fama-French factor files can be downloaded and prepared using the preprocessing utilities included in the package.

CRSP/WRDS data are not distributed with this project due to access and licensing restrictions. Users who want to reproduce the S&P 500 experiments must obtain the corresponding data through their own WRDS/CRSP access and place the required files locally inside the `data/` folder.

The `data/` folder is ignored by Git. Only a small `data/README.md` file is included to document the expected local files.

## Results

The complete experimental results are discussed in the accompanying TFG report.

The repository also stores generated CSV outputs in:

```text
results_49_Industry_Portfolios/
results_SP500_CRSP/
```

These files include summary metrics, diagnostic outputs and NAV curves used in the experimental analysis.

## Reproducibility

The experiments use fixed random seeds where stochastic methods are involved, especially in dimensionality reduction.

For exact reproducibility of the Python environment used to run the experiments, a lock file can be generated with:

```bash
pip freeze > requirements-lock.txt
```

This file records the exact package versions installed in the environment. It is mainly useful for reproducing the results of the TFG, not for publishing the package to PyPI.

## Development and packaging checks

The following commands are useful during development. They require the optional development dependencies:

```bash
pip install -e .[dev]
```

Run style checks:

```bash
python -m flake8 src scripts
```

Run a basic import check:

```bash
python -c "from tda_finance.tda.mapper_clustering import MapperParams; print(MapperParams())"
```

Build the package locally:

```bash
python -m build
```

Check the distribution before uploading to PyPI:

```bash
python -m twine check dist/*
```

## License

This project is released under the MIT License. See the `LICENSE` file for details.

