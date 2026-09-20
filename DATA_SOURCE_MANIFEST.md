# Data Source Manifest - PAPER022

Dataset: **Concrete Compressive Strength**  
Creator: **I-Cheng Yeh**  
Canonical repository: **UCI Machine Learning Repository**  
Dataset DOI: **10.24432/C5PK67**  
Foundational article DOI: **10.1016/S0008-8846(98)00165-3**  
License stated by UCI: **CC BY 4.0**

The archived analysis copy is `data/Concrete_Data.csv`. It contains 1,030 rows, eight predictors and one compressive-strength response, with no missing values. Its SHA-256 checksum is:

`27d7cf5170e65f866f7f1dcc8beabadfe8439efa083ad8cdd4764a6f75d8002e`

The working CSV was obtained from a public mirror of the canonical Yeh data and checked against the canonical UCI metadata and data structure. The manuscript cites the UCI record rather than the mirror.

## Physical mix grouping

The seven composition variables are cement, blast-furnace slag, fly ash, water, superplasticizer, coarse aggregate and fine aggregate. Exact equality on these seven variables defines the physical mix-design group. There are 427 such groups; all recorded ages belonging to one group are locked together throughout evaluation.

## Derived composition variables

- `binder = Cement + Blast + Fly Ash`
- `scm_frac = (Blast + Fly Ash) / binder`
- `slag_frac = Blast / binder`
- `fly_frac = Fly Ash / binder`
- `wb = Water / binder`
- `sp_b = Superplasticizer / binder`
- `ca_b = Coarse Aggregate / binder`
- `fa_b = Fine Aggregate / binder`

The descriptive SCM regimes are defined only from recorded slag/fly-ash presence: **No-SCM**, **Slag-only**, **Fly-ash-only**, and **Dual-SCM**. These labels make no claim about cement mineralogy. No response value is used to define a shift scenario or the composition-support diagnostic.
