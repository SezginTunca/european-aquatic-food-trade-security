# European Aquatic Food Trade Security: Replication Codebase

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24085138.svg)](https://doi.org/10.5281/zenodo.24085138)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)

This repository contains the complete empirical replication pipeline, econometric models, machine learning architectures, and high-resolution figure generation scripts for the study:

> **Continental Trade Specialization Sustains European Aquatic Food Security Beyond Local Ecological Boundaries**  
> **Author:** Sezgin Tunca  
> *Frontiers in Sustainable Food Systems* (2026)  
> **Zenodo Archive:** [https://zenodo.org/uploads/24085138](https://zenodo.org/uploads/24085138) (DOI: `10.5281/zenodo.24085138`)  
> **GitHub Repository:** [https://github.com/SezginTunca/european-aquatic-food-trade-security](https://github.com/SezginTunca/european-aquatic-food-trade-security)

---

## 1. Study Overview

European aquatic food security is structurally sustained through a continental division of labor that dissociates domestic consumption from local ecological production frontiers. Utilizing over six decades of harmonized customs microdata (UN Comtrade, CEPII BACI, 1962–2025) across 32 European sovereign nations and 16 dominant commercial species groups, this study models:

1. **Macroeconomic and Commodity Transitions:** Long-run volume/value decoupling and unit value divergence across 4 processing stages (Live/Fresh/Chilled, Frozen, Fillets, Prepared/Preserved).
2. **Specialization and Intra-Industry Trade:** Revealed comparative advantage (Balassa RCA) and Grubel-Lloyd / Greenaway-Hine-Milner (GHM) intra-industry trade decomposition.
3. **Gravity Network Architectures:** Structural Pseudo-Poisson Maximum Likelihood (PPML) gravity modeling and extreme gradient boosted trees (XGBoost / Random Forest) with TreeSHAP feature attributions.
4. **Food Safety Risk Architecture:** Poisson and negative binomial hazard models quantifying European Commission RASFF intercept risks and biological trophic biomagnification.
5. **Nutritional Delivery and Consumer Buffering:** Two-stage econometric volatility modeling of national per-capita fish supply, digestible protein, and bioavailable EPA+DHA provision per dollar.
6. **Ecological Harvest Pressure Displacement:** Trophic-level accounting ($TL \in [2.0, 4.5]$) and Live Weight Equivalent (LWE) conversion demonstrating regional ecological footprint offshoring.
7. **Constrained Markowitz-Pareto Portfolio Optimization:** Quadratic programming identifying optimal species-supplier procurement allocations that minimize supply volatility and food-safety hazard exposure under nutritional and budgetary constraints.

---

## 2. Repository Architecture

```
european-aquatic-food-trade-security/
├── .zenodo.json             <- Zenodo metadata and DOI configuration
├── CITATION.cff             <- Machine-readable citation metadata
├── LICENSE                  <- MIT open-source license
├── README.md                <- Comprehensive project documentation
├── requirements.txt         <- Python dependencies
├── run_all.sh               <- Master replication bash orchestrator
├── config/                  <- Geographic definitions, species taxonomies, and GIS geometries
│   ├── european_country_mapping.csv
│   ├── ne_50m_euro_med_countries.geojson
│   ├── sitc_commodity_taxonomy.csv
│   └── species_taxonomy_master.csv
├── data/                    <- Data directory (raw microdata excluded per data governance)
│   └── README.md            <- Detailed custodian links and download instructions
├── tables/                  <- Core analytical matrices and optimization result tables
│   ├── Table_1_European_Seafood_Macro_Panel_1962_2025.csv
│   ├── Table_1B_Processing_Premium_Scissors_1988_2024.csv
│   ├── Table_2_Species_Level_Trade_and_Unit_Values_1988_2025.csv
│   ├── Table_4_Balassa_RCA_and_Specialization.csv
│   ├── Table_5_Grubel_Lloyd_IIT_and_Quality_Decomposition.csv
│   ├── Table_9_ML_vs_Econometric_Gravity_Comparison.csv
│   ├── Table_10_SHAP_Feature_Importance.csv
│   ├── Table_A_Hazard_Notifications_by_Class_Origin.csv
│   ├── Table_B_Per_Capita_Supply_Decomposition.csv
│   ├── Table_C_Displaced_Harvest_LWE.csv
│   ├── Table_Top20_Partner_Sourcing_Matrix.csv
│   ├── Table_Opt1_Species_Model_Parameters.csv
│   ├── ...
│   └── Table_Opt8_Supplier_Archetype_Correlation_and_Shares.csv
├── figures/                 <- Output directory for publication figures (600 DPI PNG & vector PDF)
└── scripts/                 <- Sequenced, modular analysis and visualization scripts (01-14)
```

---

## 3. Sequenced Pipeline Scripts (`scripts/`)

Each analysis step is encapsulated in an independent, ordered script:

| Script | Purpose & Methodological Scope |
| :--- | :--- |
| [`01_data_extraction_and_filtering.py`](scripts/01_data_extraction_and_filtering.py) | Harmonizes multi-decade bilateral customs microdata (UN Comtrade HS-6 & BACI FOB), maps species groups, and creates derived parquet panels. |
| [`02_compute_eu_econometric_indices.py`](scripts/02_compute_eu_econometric_indices.py) | Calculates sovereign Balassa RCA, Grubel-Lloyd / GHM intra-industry trade shares, unit values, and Herfindahl-Hirschman concentration (HHI). |
| [`03_network_centrality_and_modularity.py`](scripts/03_network_centrality_and_modularity.py) | Bilateral network graph construction, Louvain community modularity, Hub/Inflow centrality, and regional clustering. |
| [`04_advanced_ml_gravity_and_simulations.py`](scripts/04_advanced_ml_gravity_and_simulations.py) | Trains structural PPML and gradient-boosted gravity models (XGBoost, Random Forest) with TreeSHAP interpretability and decadal shock simulations. |
| [`05_hazard_notifications.py`](scripts/05_hazard_notifications.py) | Models European Commission RASFF food safety notifications via Poisson and negative binomial GLMs across species and trophic levels. |
| [`06_consumer_access.py`](scripts/06_consumer_access.py) | Two-stage regression of national per-capita aquatic food consumption on import dependency, price stability, and GDP per capita. |
| [`07_harvest_pressure.py`](scripts/07_harvest_pressure.py) | Quantifies displaced marine primary harvest pressure, live-weight equivalents (LWE), and sovereign autarky deficits. |
| [`08_nutrient_supply.py`](scripts/08_nutrient_supply.py) | Computes digestible aquatic protein and long-chain omega-3 (EPA+DHA) density delivered per consumer dollar across species. |
| [`09_partner_matrix.py`](scripts/09_partner_matrix.py) | Computes the top 20 extra-European partner sourcing matrix and sovereign dependency indices. |
| [`10_extra_european_gravity.py`](scripts/10_extra_european_gravity.py) | Structural PPML gravity estimation restricted to extra-European trade corridors (Norway, Iceland, China, Ecuador, etc.). |
| [`11_mirror_check.py`](scripts/11_mirror_check.py) | Verifies BACI CIF/FOB mirror trade discrepancy ratios to validate customs reporting consistency. |
| [`12_markowitz_pareto_portfolio_optimization.py`](scripts/12_markowitz_pareto_portfolio_optimization.py) | Solves constrained Markowitz and Pareto frontiers across 16 species and 8 supplier archetypes, calculating Karush-Kuhn-Tucker (KKT) shadow prices. |
| [`13_generate_main_publication_figures.py`](scripts/13_generate_main_publication_figures.py) | Generates all 8 advanced multi-panel main publication figures (Figures 1–8) at 600 DPI publication standards with Okabe-Ito colorblind palettes. |
| [`14_generate_supplementary_figures.py`](scripts/14_generate_supplementary_figures.py) | Generates all 12 supplementary figures (Figures S1–S12) including sensitivity sweeps, diagnostics, and spatial partner matrices. |

---

## 4. Installation & Environment Setup

### Prerequisites
- Python 3.10 or higher
- Git

### Setup
Clone the repository and install dependencies using `pip` or `uv`:

```bash
# Clone repository
git clone https://github.com/SezginTunca/european-aquatic-food-trade-security.git
cd european-aquatic-food-trade-security

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

---

## 5. Replication Instructions

### Option A: Immediate Figure & Optimization Replication (Pre-computed Tables)
The repository includes all pre-computed analytical summary tables in `tables/`, allowing complete replication of all manuscript figures and portfolio optimizations without requiring 100+ GB of raw bilateral customs microdata:

```bash
# Execute portfolio optimizations and render all main & supplementary figures:
./run_all.sh --figures
```
Outputs will be saved directly into `figures/` (as 600 DPI `.png` and publication-ready vector `.pdf`) and `tables/`.

### Option B: Full Pipeline from Raw Microdata
To replicate the full econometric and machine learning pipeline from raw statistical records:
1. Download raw UN Comtrade and FAO microdata files as specified in [`data/README.md`](data/README.md) and place them into `data/raw/`.
2. Run the master orchestrator:
```bash
./run_all.sh --all
```

---

## 6. Data Governance & Access

In accordance with open-science standards and repository size limits, raw bilateral microdata files (>100 GB) are **not** bundled directly in this repository. All primary data are accessible from their respective public custodians:

- **UN Comtrade Database:** [https://comtradeplus.un.org/](https://comtradeplus.un.org/)
- **CEPII BACI World Trade Database:** [http://www.cepii.fr/CEPII/en/bdd_modele/bdd_modele_item.asp?id=37](http://www.cepii.fr/CEPII/en/bdd_modele/bdd_modele_item.asp?id=37)
- **FAO Fisheries and Aquaculture Statistics (FishStatJ & FAOSTAT):** [https://www.fao.org/fishery/en/statistics](https://www.fao.org/fishery/en/statistics)
- **European Commission Rapid Alert System for Food and Feed (EC RASFF):** [https://food.ec.europa.eu/food-safety/rasff_en](https://food.ec.europa.eu/food-safety/rasff_en)
- **FishBase & SeaLifeBase:** [https://www.fishbase.org/](https://www.fishbase.org/)

Detailed field mapping and download queries are documented in [`data/README.md`](data/README.md).

---

## 7. Citation

If you use this codebase, empirical methodology, or analytical matrices in your research, please cite:

```bibtex
@article{tunca2026continental,
  author    = {Tunca, Sezgin},
  title     = {Continental Trade Specialization Sustains European Aquatic Food Security Beyond Local Ecological Boundaries},
  journal   = {Frontiers in Sustainable Food Systems},
  year      = {2026},
  doi       = {10.5281/zenodo.24085138},
  url       = {https://github.com/SezginTunca/european-aquatic-food-trade-security}
}
```

---

## 8. License

This project is licensed under the terms of the [MIT License](LICENSE).
