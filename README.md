# PAPER022 - Public reproducibility repository

Public reproducibility materials for the manuscript:

**Calibration Transfer of Conformal Prediction for Concrete Strength Across SCM Composition Regimes**

Author: **Sunilgar L. Gusai**  
Target journal: *Journal of Computers, Mechanical and Management (JCMM)*

This repository accompanies PAPER022 and contains the analysis code, fixed-seed result summaries, environment information, data-source documentation, and verification logs used for the journal revision.

The source dataset is the Concrete Compressive Strength dataset from the UCI Machine Learning Repository (DOI: 10.24432/C5PK67).

The archived experiment environment used Python 3.13.5, NumPy 2.3.5, pandas 2.2.3, scikit-learn 1.8.0, and Matplotlib 3.10.8.

Primary seeds: 20260901-20260930. Random-Forest sensitivity seeds: 20260901-20260910.

All curing ages of the same seven-component physical mix design are kept in one partition to prevent cross-age mix leakage.

Custom analysis code is released under the MIT License. The source dataset retains its original CC BY 4.0 terms.
