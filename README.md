# PAPER022 — Reproducibility Repository

## Calibration Transfer of Conformal Prediction for Concrete Strength Across SCM Composition Regimes

**Author:** Sunilgar L. Gusai  
**ORCID:** https://orcid.org/0009-0004-0739-4812  
**Target journal:** *Journal of Computers, Mechanical and Management (JCMM)*  
**Project ID:** PAPER022  
**Repository:** https://github.com/SunilgarGusai/PAPER-JCMM-reproducibility

This repository provides the public reproducibility materials accompanying the manuscript **“Calibration Transfer of Conformal Prediction for Concrete Strength Across SCM Composition Regimes.”** The study examines how conformal prediction behaves when concrete-strength models are transferred across physically interpretable supplementary cementitious material (SCM) composition shifts.

The repository is intended to make the computational workflow transparent and auditable. It contains the analysis code, sensitivity-analysis code, fixed-seed design information, machine-readable result summaries, figure-generation scripts, data-source documentation, licensing information, and numerical verification logs used for the journal revision.

## Study design in brief

The analysis uses the Concrete Compressive Strength dataset distributed by the UCI Machine Learning Repository. The 1,030 observations correspond to **427 distinct seven-component physical mix designs**. To prevent cross-age mix leakage, all curing ages belonging to the same physical mixture are kept in the same partition.

The evaluation includes:

- a grouped reference setting;
- graded SCM-composition enrichment shifts;
- complete holdout of SCM composition regimes;
- a 28-day-only control analysis;
- ordinary 90% split conformal prediction;
- estimated weighted conformal prediction under covariate shift;
- composition-space support diagnostics;
- a fixed Random-Forest sensitivity analysis.

The paper treats finite interval width, unbounded intervals, effective sample size, support, and empirical coverage as complementary deployment diagnostics rather than reporting coverage alone.

## Repository structure

```text
.
├── code/
│   ├── paper022_analysis_v2.py
│   ├── paper022_sensitivity_v2.py
│   ├── make_figures_v2.py
│   └── verify_v2_saved_results.py
├── data/
│   └── README.md
├── results_v2/
│   ├── submission_summary_v2.csv
│   ├── table2_graded_shift_R1.csv
│   ├── table3_hard_holdout_R1.csv
│   ├── model_sensitivity_random_forest_v2.csv
│   ├── model_sensitivity_random_forest_summary_v2.csv
│   ├── support_threshold_sensitivity_v2.csv
│   ├── data_audit_v2.json
│   ├── repetition_counts_v2.csv
│   └── environment_and_design_v2.json
├── logs/
│   └── v2_sentinel_reproduction_audit.json
├── DATA_SOURCE_MANIFEST.md
├── DATA_LICENSE.md
├── README_REPRODUCIBILITY.md
├── CITATION.cff
├── requirements.txt
└── LICENSE
```

The exact analysis CSV and the complete primary replicate-level results are included in the reproducibility ZIP submitted with the manuscript. They can be regenerated from the public code after obtaining the canonical UCI dataset as described in `data/README.md`.

## Data source

**Concrete Compressive Strength**  
UCI Machine Learning Repository  
DOI: https://doi.org/10.24432/C5PK67

The source dataset is distributed under **CC BY 4.0**. The dataset is not relicensed by this repository. Source, checksum, attribution, and derived-variable details are documented in `DATA_SOURCE_MANIFEST.md` and `DATA_LICENSE.md`.

## Computational environment

The archived experiment environment used:

- Python 3.13.5
- NumPy 2.3.5
- pandas 2.2.3
- scikit-learn 1.8.0
- Matplotlib 3.10.8

Pinned package versions are listed in `requirements.txt`.

Primary repetitions use the fixed seeds **20260901–20260930**. The Random-Forest sensitivity analysis uses **20260901–20260910**. No seed was selected or discarded according to its result.

## Reproducing the analysis

1. Clone the repository.
2. Download the canonical UCI dataset and prepare `data/Concrete_Data.csv` as documented in `data/README.md`.
3. Create a clean Python environment and install the pinned dependencies.
4. Run the primary analysis, sensitivity analysis, figure generator, and verification script.

```bash
git clone https://github.com/SunilgarGusai/PAPER-JCMM-reproducibility.git
cd PAPER-JCMM-reproducibility

python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
python code/paper022_analysis_v2.py
python code/paper022_sensitivity_v2.py
python code/make_figures_v2.py
python code/verify_v2_saved_results.py
```

## Verification status

The archived verification audit records:

- 1,030 observations;
- 992 unique eight-predictor configurations;
- 427 unique seven-component mix designs;
- 24 primary scenario × method cells;
- 30 repetitions in every primary cell;
- all sentinel comparisons passed;
- maximum absolute stored-versus-regenerated discrepancy: `1.7763568394002505e-15`.

Every headline numerical result in the revised manuscript is traceable to machine-readable outputs. Scientific figures are generated programmatically from those outputs and are not manually redrawn.

## Citation

Citation metadata are provided in `CITATION.cff`. If these materials are reused, please cite both the associated manuscript and the original UCI dataset.

Until the journal article receives its final bibliographic record, cite the repository as:

> Gusai, S. L. (2026). *Reproducibility files for Calibration Transfer of Conformal Prediction for Concrete Strength Across SCM Composition Regimes*. GitHub repository. https://github.com/SunilgarGusai/PAPER-JCMM-reproducibility

## Licence

The custom analysis code in this repository is released under the **MIT License**. The UCI source dataset retains its **CC BY 4.0** licence and attribution requirements.

## Contact

**Dr. Sunilgar L. Gusai**  
Faculty of Computer Applications, Marwadi University, Rajkot, Gujarat, India  
ORCID: https://orcid.org/0009-0004-0739-4812

## Academic profile and related research

This repository is part of the open-research programme of **Dr. Sunilgar L. Gusai**, spanning spectral graph theory, network science and reproducible computational modelling.

- Academic portfolio: https://sunilgargusai.github.io/sunilgar-portfolio/
- GitHub profile: https://github.com/SunilgarGusai
- ORCID: https://orcid.org/0009-0004-0739-4812
- Related VELE / power-grid repository: https://github.com/SunilgarGusai/VELE-PowerGrid-Reproducibility
