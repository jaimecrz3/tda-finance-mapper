Experiments
===========

The final experiments can be run from the project root with:

.. code-block:: bash

   python -m tda_finance.experiments.run_mapper_ph_experiments

The experiment script compares three strategies:

1. Mapper;
2. Mapper with persistent-homology regime control;
3. equal-weight benchmark.

The selected dataset is configured inside the script.

Data preparation
----------------

Auxiliary scripts used to prepare the S&P 500 CRSP data are located in:

.. code-block:: text

   scripts/data_preparation/

Example:

.. code-block:: bash

   python scripts/data_preparation/prepare_prices_sp500.py
   python scripts/data_preparation/make_monthly_sp500.py

These scripts are included to make the data preparation process more
transparent, but they are not part of the main package API.

Data availability
----------------

The repository does not include the raw financial datasets used in the experiments.

The Kenneth French 49 Industry Portfolios and Fama-French factor files can be downloaded and prepared using the preprocessing utilities included in the package.

CRSP/WRDS data are not distributed with this project due to access and licensing restrictions. Users who want to reproduce the S&P 500 experiments must obtain the corresponding data through their own WRDS/CRSP access and place the required files locally inside the ``data/`` folder.

.. note::
   The ``data/`` folder is ignored by Git. Only a small ``data/README.md`` file is included to document the expected local files.

Experimental results
--------------------

The final experiments generate CSV files with summary metrics, NAV curves and
persistent-homology diagnostics.

The main output folders are:

.. code-block:: text

   results_49_Industry_Portfolios/
   results_SP500_CRSP/

The complete interpretation of the results is provided in the accompanying TFG
report. In summary, the experiments compare Mapper, Mapper with persistent
homology regime control and the equal-weight benchmark under the same causal
backtesting protocol.

The documentation included here focuses on the code and reproducibility of the
experiments, while the academic report contains the full discussion of the
financial results.
