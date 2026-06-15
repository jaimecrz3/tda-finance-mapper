Usage
=====

The following example shows the basic use of the package API with a generic
price matrix.

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

This is only a minimal usage example. The complete experimental protocol is
implemented in the experiment script.
