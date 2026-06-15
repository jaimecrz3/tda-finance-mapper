# Data folder

This folder is intentionally not versioned.

The project uses financial datasets that must be obtained or generated locally by the user.

## Kenneth French data

The Kenneth French 49 Industry Portfolios and Fama-French factor files can be downloaded and prepared using the preprocessing utilities included in the package.

Expected generated files include:

```text
data/49_industries_portfolios_monthly.csv
data/ff3_monthly.csv
```

## CRSP / WRDS Data

CRSP/WRDS data are not included in this repository due to access and licensing restrictions.

Users who want to reproduce the S&P 500 experiments must obtain the corresponding data through their own WRDS/CRSP access and prepare the local files required by the preprocessing scripts.
