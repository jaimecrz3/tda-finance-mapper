User manual
===========

This manual explains how to use ``tda-finance-mapper`` from installation to
obtaining portfolio backtesting results. It is intended for users who want to
run their own experiments without needing to inspect the internal source code.

Overview
--------

``tda-finance-mapper`` is a Python package for applying Topological Data
Analysis (TDA) to financial time series. The package focuses on three main
tasks:

* building Mapper-based portfolio strategies;
* adding a persistent-homology regime-control signal;
* comparing the resulting strategies with an equal-weight benchmark.

The package is used as a Python library. This means that the user does not
"open" the package as a standalone application. Instead, the user writes a
Python script, imports the functions provided by ``tda_finance`` and applies
them to a price matrix.

There are two common ways to use the project:

* **Option A:** install the package from PyPI and run your own experiment with
  your own data.
* **Option B:** clone the repository and reproduce or adapt the dissertation
  experiments.

For new datasets, Option A is usually the simplest starting point.

The package is intended for academic and experimental use. It should not be
interpreted as investment advice.

Option A: use the package with your own data
--------------------------------------------

This option is recommended for users who want to run their own experiment
without modifying the source code of the package.

Step 1: create a working folder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a folder for your experiment. For example:

.. code-block:: text

   my_tda_experiment/
   ├── data/
   │   └── input_data.csv
   ├── results/
   └── run_my_experiment.py

The file ``input_data.csv`` will contain your input data. The file
``run_my_experiment.py`` will contain the Python code that loads the data,
runs the strategies and saves the results.

All commands in this section should be executed from the working folder
``my_tda_experiment/``.

Step 2: create a virtual environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It is recommended to use a virtual environment so that the package and its
dependencies do not interfere with other Python projects.

From the working folder, run:

.. code-block:: bash

   python -m venv .venv

On Windows, activate it with:

.. code-block:: bash

   .venv\Scripts\activate

On macOS or Linux, activate it with:

.. code-block:: bash

   source .venv/bin/activate

Step 3: install the package
~~~~~~~~~~~~~~~~~~~~~~~~~~~

With the virtual environment activated, install the package from PyPI:

.. code-block:: bash

   pip install tda-finance-mapper

Check that the package has been installed correctly:

.. code-block:: bash

   python -c "import tda_finance; print('tda_finance imported successfully')"

Step 4: prepare your input data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The backtesting functions expect a price matrix. This means that, before
calling ``backtest_tda`` or ``backtest_equal_weight_rebalanced``, the data
passed to the package must represent prices or a price index.

However, financial datasets are often available in two different formats:

* prices;
* returns.

Both cases can be used, but returns must be converted into a price index before
running the backtest.

For this reason, in this manual the input file is called
``data/input_data.csv``.

Case 1: the CSV contains prices
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Example ``data/input_data.csv``:

.. code-block:: text

   date,AAPL,MSFT,JPM
   2015-01-31,100.0,100.0,100.0
   2015-02-28,101.2,98.7,102.5
   2015-03-31,99.8,97.1,104.0
   2015-04-30,103.4,99.5,105.2

In this case, the data can be used directly as prices.

Case 2: the CSV contains returns
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Some financial datasets contain returns instead of prices. For example, the
49 Industry Portfolios are commonly provided as monthly returns.

Example ``data/input_data.csv`` with decimal returns:

.. code-block:: text

   date,Agric,Food,Soda
   1970-01-01,0.0101,-0.0277,-0.0279
   1970-02-01,0.0993,0.0601,0.0388
   1970-03-01,-0.1339,-0.0056,-0.0110

Here, ``0.0101`` means ``1.01%``. These returns must be converted into a price
index before running the backtest:

.. code-block:: python

   prices = (1.0 + returns).cumprod() * 100.0

If returns are expressed as percentages, for example ``1.01`` instead of
``0.0101``, they must first be divided by 100:

.. code-block:: python

   returns = returns / 100.0
   prices = (1.0 + returns).cumprod() * 100.0

Important: the short tables above are only illustrations of the required
format. A real backtest needs enough observations. For example, with
``lookback_days=60`` and monthly data, the strategy needs at least 60 months of
history before the first rebalance.

Step 5: create the experiment script
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a file called ``run_my_experiment.py`` in your working folder:

.. code-block:: text

   my_tda_experiment/
   ├── data/
   │   └── input_data.csv
   ├── results/
   └── run_my_experiment.py

Paste the following code into ``run_my_experiment.py``:

.. code-block:: python

   import os

   import pandas as pd

   from tda_finance.portfolio.backtest_engine import (
       backtest_equal_weight_rebalanced,
       backtest_tda,
       perf_summary,
   )
   from tda_finance.tda.mapper_clustering import MapperParams

   # -------------------------------------------------------------------------
   # 1. Load input data
   # -------------------------------------------------------------------------

   DATA_FILE = "data/input_data.csv"

   # Set this to True if the CSV contains returns.
   # Set this to False if the CSV already contains prices.
   DATA_ARE_RETURNS = True

   # Set this to True if returns are written as percentages, for example 1.01.
   # Set this to False if returns are written as decimals, for example 0.0101.
   RETURNS_ARE_PERCENTAGES = False

   data = pd.read_csv(
       DATA_FILE,
       index_col=0,
       parse_dates=True,
   )

   data = data.sort_index()
   data = data.astype(float)

   print("Loaded input data:")
   print(data.head())
   print("Input shape:", data.shape)

   if DATA_ARE_RETURNS:
       returns = data.copy()

       if RETURNS_ARE_PERCENTAGES:
           returns = returns / 100.0

       if (returns <= -1.0).any().any():
           raise ValueError(
               "Some returns are lower than or equal to -100%. "
               "Check whether the input data are really returns."
           )

       # The package expects prices. Returns are converted into a price index
       # starting at 100.
       prices = (1.0 + returns).cumprod() * 100.0

   else:
       prices = data.copy()

       if (prices <= 0.0).any().any():
           raise ValueError(
               "Some prices are lower than or equal to zero. "
               "Check the input price matrix."
           )

   print("\nPrice matrix used by the backtest:")
   print(prices.head())
   print("Price shape:", prices.shape)

   # -------------------------------------------------------------------------
   # 2. Define Mapper parameters
   # -------------------------------------------------------------------------

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

   # -------------------------------------------------------------------------
   # 3. Run Mapper
   # -------------------------------------------------------------------------

   mapper_result = backtest_tda(
       prices=prices,
       lookback_days=60,
       rebalance_days=3,
       params=params,
       tc_bps=5.0,
       use_ph_control=False,
   )

   mapper_metrics = perf_summary(
       mapper_result["port_ret"],
       periods_per_year=12,
   )

   # -------------------------------------------------------------------------
   # 4. Run Mapper with PH regime control
   # -------------------------------------------------------------------------

   mapper_ph_result = backtest_tda(
       prices=prices,
       lookback_days=60,
       rebalance_days=3,
       params=params,
       tc_bps=5.0,
       use_ph_control=True,
       regime_action="equal_weight",
   )

   mapper_ph_metrics = perf_summary(
       mapper_ph_result["port_ret"],
       periods_per_year=12,
   )

   # -------------------------------------------------------------------------
   # 5. Run equal-weight benchmark
   # -------------------------------------------------------------------------

   eqw_result = backtest_equal_weight_rebalanced(
       prices=prices,
       lookback_days=60,
       rebalance_days=3,
       tc_bps=5.0,
   )

   eqw_metrics = perf_summary(
       eqw_result["port_ret"],
       periods_per_year=12,
   )

   # -------------------------------------------------------------------------
   # 6. Save results
   # -------------------------------------------------------------------------

   os.makedirs("results", exist_ok=True)

   comparison = pd.DataFrame(
       [
           {"model": "Mapper", **mapper_metrics},
           {"model": "Mapper + PH", **mapper_ph_metrics},
           {"model": "Equal-weight", **eqw_metrics},
       ]
   )

   comparison.to_csv("results/comparison_metrics.csv", index=False)

   nav_curves = pd.DataFrame(
       {
           "Mapper": mapper_result["port_nav"],
           "Mapper + PH": mapper_ph_result["port_nav"],
           "Equal-weight": eqw_result["port_nav"],
       }
   )

   nav_curves.to_csv("results/nav_curves.csv")

   mapper_result.to_csv("results/mapper_result.csv")
   mapper_ph_result.to_csv("results/mapper_ph_result.csv")
   eqw_result.to_csv("results/equal_weight_result.csv")

   print("\nPerformance comparison:")
   print(comparison)

   print("\nResults saved in the 'results/' folder.")
   
In the examples of this manual, ``periods_per_year=12`` is used because the
data are assumed to be monthly. For daily data, this value should normally be
changed to ``252``.

Step 6: run the experiment
~~~~~~~~~~~~~~~~~~~~~~~~~~

From the working folder, run:

.. code-block:: bash

   python run_my_experiment.py

After execution, the folder should contain:

.. code-block:: text

   my_tda_experiment/
   ├── data/
   │   └── input_data.csv
   ├── results/
   │   ├── comparison_metrics.csv
   │   ├── nav_curves.csv
   │   ├── mapper_result.csv
   │   ├── mapper_ph_result.csv
   │   └── equal_weight_result.csv
   └── run_my_experiment.py

Step 7: interpret the output files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The file ``comparison_metrics.csv`` contains one row per model and one column
per metric.

Main metrics:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Metric
     - Interpretation
   * - ``total_return``
     - Total return over the full evaluation period.
   * - ``ann_return_geo``
     - Geometric annualized return.
   * - ``ann_vol``
     - Annualized volatility.
   * - ``sharpe``
     - Return adjusted by volatility.
   * - ``sortino``
     - Return adjusted by downside volatility.
   * - ``max_dd``
     - Maximum drawdown.

The file ``nav_curves.csv`` contains the cumulative value of each strategy.
The NAV starts at 1. A final NAV greater than 1 means that the strategy has
increased in value over the evaluation period.

A higher final NAV indicates higher cumulative return. However, NAV should not
be interpreted alone. It should be compared with volatility, Sharpe, Sortino
and maximum drawdown.

Step 8: plot the NAV curves
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create another file called ``plot_results.py`` in the same working folder:

.. code-block:: text

   my_tda_experiment/
   ├── data/
   │   └── input_data.csv
   ├── results/
   │   └── nav_curves.csv
   ├── run_my_experiment.py
   └── plot_results.py

Paste the following code into ``plot_results.py``:

.. code-block:: python

   import pandas as pd
   import matplotlib.pyplot as plt

   nav = pd.read_csv(
       "results/nav_curves.csv",
       index_col=0,
       parse_dates=True,
   )

   nav.plot()
   plt.title("Portfolio NAV comparison")
   plt.xlabel("Date")
   plt.ylabel("NAV")
   plt.tight_layout()
   plt.show()

Run it with:

.. code-block:: bash

   python plot_results.py

This will display a graph comparing the cumulative evolution of Mapper,
Mapper with PH regime control and the equal-weight benchmark.

Option B: use the repository
----------------------------

This option is useful for users who want to inspect the source code, modify the
package or reproduce the dissertation experiments.

Step 1: clone the repository
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/jaimecrz3/tda-finance-mapper.git
   cd tda-finance-mapper

Step 2: create and activate a virtual environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python -m venv .venv

On Windows:

.. code-block:: bash

   .venv\Scripts\activate

On macOS or Linux:

.. code-block:: bash

   source .venv/bin/activate

Step 3: install the package in editable mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install -e .

To install the dependencies listed in the requirements file:

.. code-block:: bash

   pip install -r requirements.txt

Editable mode is useful during development because changes made to the source
code are available without reinstalling the package.

Step 4: use your own data from the repository
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A user can also run custom experiments from inside the repository. For example:

.. code-block:: text

   tda-finance-mapper/
   ├── data/
   │   └── input_data.csv
   ├── results/
   └── scripts/
       └── run_my_experiment.py

The content of ``scripts/run_my_experiment.py`` can be the same as in Option A.
The command should be executed from the root of the repository:

.. code-block:: bash

   python scripts/run_my_experiment.py

This is important because the path ``data/input_data.csv`` is interpreted relative
to the folder from which the command is executed.

Reproducing the dissertation experiments
----------------------------------------

The repository includes an experiment script designed to reproduce the final
experiments presented in the dissertation.

From the root of the repository, run:

.. code-block:: bash

   python -m tda_finance.experiments.run_mapper_ph_experiments

This script compares:

* Mapper;
* Mapper with PH regime control;
* equal-weight benchmark.

The script is specific to the datasets used in the dissertation:

* 49 Industry Portfolios from the Kenneth R. French Data Library;
* S&P 500/CRSP data.

Therefore, this script is not intended as a generic entry point for arbitrary
datasets. For new datasets, prepare a price matrix and use the workflow from
Option A.

Input data details
------------------

Price matrix
~~~~~~~~~~~~

The backtesting functions expect prices or a price index. If the original CSV contains returns, 
they must be converted into a price index before calling the backtesting functions.

Correct format:

.. code-block:: text

   date,Asset_1,Asset_2,Asset_3
   2015-01-31,100.0,100.0,100.0
   2015-02-28,101.2,98.7,102.5
   2015-03-31,99.8,97.1,104.0

The first column should be parsed as dates and used as the index.

Useful checks:

.. code-block:: python

   print(prices.head())
   print(prices.tail())
   print(prices.shape)
   print(prices.index)
   print(prices.isna().sum().sort_values(ascending=False).head())

Missing values
~~~~~~~~~~~~~~

Inside each historical window, the package applies forward filling and then
removes assets that still contain missing values in that window. This avoids
using future information to fill past observations.

If many assets have missing values, the available universe may become small in
some rebalance dates. In that case, inspect the dataset before running the
experiment.

Monthly and daily data
~~~~~~~~~~~~~~~~~~~~~~

The package works with a generic time index. However, the interpretation of
``lookback_days`` and ``rebalance_days`` depends on the frequency of the data.

In the dissertation experiments, monthly data are used. Therefore:

* ``lookback_days=60`` means 60 monthly observations;
* ``rebalance_days=3`` means rebalance every 3 monthly observations.

The parameter names use ``days`` because the same functions can be applied to
other frequencies, but the values should be interpreted as number of
observations.

Main parameters
---------------

Mapper parameters
~~~~~~~~~~~~~~~~~

Mapper parameters are configured through ``MapperParams``.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Parameter
     - Meaning
   * - ``pca_var``
     - Explained variance retained by PCA before applying UMAP.
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
     - Linkage criterion for agglomerative clustering.
   * - ``dbscan_eps``
     - Radius parameter used by DBSCAN if ``clusterer="dbscan"``.
   * - ``dbscan_min_samples``
     - Minimum number of samples used by DBSCAN.
   * - ``random_state``
     - Random seed used for reproducibility.
   * - ``min_assets``
     - Minimum number of assets required in a window.

Backtesting parameters
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Parameter
     - Meaning
   * - ``lookback_days``
     - Number of observations used as historical window.
   * - ``rebalance_days``
     - Number of observations between two consecutive rebalances.
   * - ``tc_bps``
     - Transaction cost in basis points, applied proportionally to turnover.
   * - ``use_ph_control``
     - Whether to activate persistent-homology regime control.
   * - ``regime_action``
     - Defensive action when PH flags an anomalous regime. Supported values are
       ``cash`` and ``equal_weight``.
   * - ``weight_method``
     - Rule used to transform Mapper clusters into portfolio weights. Supported
       values are ``node_overlap`` and ``macro_cluster``.

Understanding the strategies
----------------------------

Mapper
~~~~~~

The Mapper strategy builds a point cloud from recent asset returns. Each point
represents one asset, and its coordinates are the returns observed in the
historical window. Mapper is then applied to obtain a graph, and this graph is
converted into portfolio weights.

Mapper with PH regime control
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Mapper + PH strategy uses Mapper as the main portfolio construction method.
In addition, it computes a persistent-homology regime score from a recent
window of returns.

When the PH signal flags an anomalous regime, the strategy applies the selected
defensive action. In the examples above, the defensive action is
``equal_weight``.

Equal-weight
~~~~~~~~~~~~

The equal-weight benchmark assigns the same weight to all available assets at
each rebalance date. It is used as a simple and transparent benchmark.

Common problems
---------------

``ModuleNotFoundError: No module named 'tda_finance'``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The package is probably not installed in the current environment. Activate the
correct virtual environment and install the package again.

From PyPI:

.. code-block:: bash

   pip install tda-finance-mapper

From the repository:

.. code-block:: bash

   pip install -e .

``Time series is too short for the selected lookback``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The dataset has too few observations for the selected ``lookback_days``. Use a
shorter lookback window or provide a longer price history.

For example, with monthly data and ``lookback_days=60``, the strategy needs at
least 60 months of history before the first rebalance.

Wrong path to ``input_data.csv``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If Python cannot find ``data/input_data.csv``, check the folder from which the
script is being executed.

For Option A, execute:

.. code-block:: bash

   python run_my_experiment.py

from the folder:

.. code-block:: text

   my_tda_experiment/

For Option B, execute:

.. code-block:: bash

   python scripts/run_my_experiment.py

from the repository root:

.. code-block:: text

   tda-finance-mapper/

Unexpected results
~~~~~~~~~~~~~~~~~~

Check that:

* dates are correctly parsed;
* prices are sorted by date;
* columns represent assets;
* the matrix passed to the backtesting functions represents prices or a price index;
* if the original CSV contains returns, they have been converted correctly;
* the rebalance frequency matches the data frequency;
* transaction costs are set as intended;
* the dataset contains enough observations for the selected lookback window.

Financial disclaimer
--------------------

This package is intended for academic and experimental use. The results should
not be interpreted as investment advice. Any operational use would require
additional validation, liquidity analysis, realistic transaction-cost modelling
and broader out-of-sample testing.
