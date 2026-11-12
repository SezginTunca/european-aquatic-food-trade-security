# Data Acquisition and Directory Architecture

Per open-science data governance guidelines and storage limits, raw microdata files are **not** bundled directly into this git repository. All primary datasets utilized in this study are public administrative and statistical records available from official international custodians:

## 1. Primary Data Custodians

| Dataset | Source / Custodian | Public Access Link | Spatial / Temporal Coverage |
| :--- | :--- | :--- | :--- |
| **Bilateral Trade Microdata** | UN Comtrade / CEPII BACI | [UN Comtrade](https://comtradeplus.un.org/) / [CEPII BACI](http://www.cepii.fr/CEPII/en/bdd_modele/bdd_modele_item.asp?id=37) | Global bilateral flows, 1988–2025 (HS-6 & species level) |
| **Longitudinal Commodity Panel** | UN Comtrade SITC Rev.1 | [UN Comtrade Bulk](https://comtradeplus.un.org/) | 1962–2025, 4 processing categories |
| **National Food Balance Sheets** | FAO Food Balance Sheets (FAOSTAT) | [FAOSTAT Food Balances](https://www.fao.org/faostat/en/#data/FBS) | European per-capita fish supply, 1961–2024 |
| **Aquaculture & Capture Production** | FAO FishStatJ | [FAO Global Production](https://www.fao.org/fishery/en/statistics) | National marine catch and farming volumes, 1950–2024 |
| **Food-Safety Hazard Surveillance** | European Commission Rapid Alert System for Food and Feed (EC RASFF) & FAO GLOBEFISH | [EC RASFF Portal](https://food.ec.europa.eu/food-safety/rasff_en) / [FAO GLOBEFISH](https://www.fao.org/in-action/globefish/en/) | Hazard notifications by country of origin, product, and pathogen/chemical class, 2015–2024 |
| **Ecological Trophic Levels** | FishBase & SeaLifeBase | [FishBase](https://www.fishbase.org/) / [SeaLifeBase](https://www.sealifebase.ca/) | Species biological trophic levels ($TL \in [2.0, 4.5]$) |
| **Nutritional Profiles** | Golden et al. (2021) / Aquatic Foods Composition Database (AFCD) | *Nature* DOI: 10.1038/s41586-021-03917-1 | Edible protein, EPA+DHA, vitamin B12, and selenium densities |
| **Macroeconomic & Demographic** | World Bank World Development Indicators (WDI) | [World Bank DataBank](https://databank.worldbank.org/) | National population and deflators, 1960–2024 |

## 2. Directory Placement

When reproducing analyses from raw sources, place files into the following paths:

```
data/
├── raw/                  <- Place raw UN Comtrade, FAO, and RASFF CSV/XLSX downloads here
└── derived/              <- Generated parquet panels from Script 01:
    ├── eu_species_bilateral_panel.parquet
    ├── eu_species_intra_panel.parquet
    ├── eu_commodity_sitc_panel_1962_2025.parquet
    └── master_global_bilateral_panel.parquet
```

Researchers seeking pre-harmonized intermediate panels can contact the corresponding author (`sezgin.tunca@gmail.com`) for direct scientific access.

