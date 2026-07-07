User manual
===========

This page provides a practical guide to using ``tda-finance-mapper``. The
package implements a reproducible pipeline for applying Topological Data
Analysis to financial data and evaluating Mapper-based portfolio strategies.

Installation
------------

The package can be installed from PyPI with:

.. code-block:: bash

   pip install tda-finance-mapper

For local development, clone the repository and install it in editable mode:

.. code-block:: bash

   pip install -e .

To install the dependencies listed in the project requirements file, run:

.. code-block:: bash

   pip install -r requirements.txt

Input data format
-----------------

The main backtesting functions, such as ``backtest_tda`` and
``backtest_equal_weight_rebalanced``, expect a price matrix as a
``pandas.DataFrame``. Rows must represent dates and columns must represent
assets.

Example:

.. code-block:: text

              AAPL     MSFT      JPM
   2020-01-31 100.0    100.0    100.0
   2020-02-29 105.2    102.7     97.8
   2020-03-31  92.4     96.3     84.1

The index should be a ``DatetimeIndex`` and the columns should contain asset
identifiers. The package computes returns internally from this price matrix
when running the backtest.

The final experiment script included in the project,
``run_mapper_ph_experiments.py``, is more specific. It is designed to reproduce
the experiments of the dissertation using the 49 Industry Portfolios and the
S&P 500/CRSP dataset. For other datasets, users must first prepare their own
price matrix in the format described above and then call the backtesting
functions directly.

Main Mapper parameters
----------------------

Mapper parameters are configured through ``MapperParams``.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Parameter
     - Meaning
   * - ``pca_var``
     - Explained variance retained by PCA before UMAP.
   * - ``umap_dim``
     - Output dimension of UMAP used to construct the lens.
   * - ``n_cubes``
     - Number of intervals in the Mapper cover.
   * - ``perc_overlap``
     - Percentage of overlap between cover intervals.
   * - ``clusterer``
     - Local clustering method. In the final experiments, ``haca`` is used.
   * - ``haca_distance_threshold``
     - Distance threshold for agglomerative clustering.
   * - ``haca_linkage``
     - Linkage criterion used by agglomerative clustering.
   * - ``random_state``
     - Random seed used for reproducibility.

Main backtesting parameters
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Parameter
     - Meaning
   * - ``lookback_days``
     - Length of the historical window used to build the portfolio.
   * - ``rebalance_days``
     - Number of observations between rebalances.
   * - ``tc_bps``
     - Transaction cost in basis points, applied proportionally to turnover.
   * - ``use_ph_control``
     - Whether to activate persistent-homology regime control.
   * - ``regime_action``
     - Defensive action when PH flags an anomalous regime. Supported values are
       ``cash`` and ``equal_weight``.

Case 1: Mapper portfolio
------------------------

The following example runs a Mapper-based portfolio strategy without
persistent-homology regime control.

.. code-block:: python

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

The output ``result`` is a ``pandas.DataFrame`` with the following columns:

.. code-block:: python

   result["port_ret"]   # portfolio returns
   result["port_nav"]   # cumulative NAV
   result["turnover"]   # portfolio turnover

Case 2: Mapper with PH regime control
-------------------------------------

The following example activates persistent-homology regime control. When the
PH signal flags an anomalous regime, the strategy switches temporarily to an
equal-weight portfolio.

.. code-block:: python

   result_ph = backtest_tda(
       prices=prices,
       lookback_days=60,
       rebalance_days=3,
       params=params,
       tc_bps=5.0,
       use_ph_control=True,
       regime_action="equal_weight",
   )

   metrics_ph = perf_summary(
       result_ph["port_ret"],
       periods_per_year=12,
   )

   print(metrics_ph)

When PH control is active, the output includes additional diagnostic columns:

.. code-block:: python

   result_ph["landscape_norm_L2"]
   result_ph["market_safe_flag"]

``landscape_norm_L2`` stores the PH-based regime score. ``market_safe_flag`` is
equal to 1 when the regime is considered safe and 0 when the PH signal flags an
anomalous regime.

Case 3: Equal-weight benchmark
------------------------------

The package also includes an equal-weight benchmark evaluated with the same
rebalance frequency and transaction-cost convention.

.. code-block:: python

   from tda_finance.portfolio.backtest_engine import (
       backtest_equal_weight_rebalanced,
       perf_summary,
   )

   eqw = backtest_equal_weight_rebalanced(
       prices=prices,
       lookback_days=60,
       rebalance_days=3,
       tc_bps=5.0,
   )

   metrics_eqw = perf_summary(
       eqw["port_ret"],
       periods_per_year=12,
   )

   print(metrics_eqw)

Case 4: Final experiment script
-------------------------------

The final experiment script compares three strategies:

1. Mapper.
2. Mapper with PH regime control.
3. Equal-weight benchmark.

From the project root, run:

.. code-block:: bash

   python -m tda_finance.experiments.run_mapper_ph_experiments

The script saves results as CSV files in the corresponding results directory.

Data requirements
-----------------

The project uses two main datasets:

* 49 Industry Portfolios from the Kenneth R. French Data Library.
* S&P 500/CRSP data.

The CRSP/WRDS data are not included in the repository because they require
institutional access or a separate license.

Financial disclaimer
--------------------

This package is intended for academic and experimental use. The results should
not be interpreted as investment advice.
