#!/usr/bin/env python3
"""
scripts/11_mirror_check.py
==========================
BACI Harmonized Mirror & CIF/FOB Robustness Benchmark for Top-20 Sourcing Corridors.

Compares European reported customs imports (UN Comtrade / Eurostat CIF) against
CEPII BACI reconciled global trade flows (FOB; 1995-2024) across the Top 20
extra-European seafood supplying economies (2015-2024).

Output:
- tables/Table_SI_BACI_Mirror_Check_Top20.csv
"""

import os
import sys
import pandas as pd
import numpy as np
import pyarrow.dataset as ds

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "derived")
TABLES_DIR = os.path.join(PROJECT_ROOT, "tables")
CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")
os.makedirs(TABLES_DIR, exist_ok=True)

BILATERAL_PARQUET = os.path.join(DATA_DIR, "eu_species_bilateral_panel.parquet")
COUNTRY_MAP_CSV = os.path.join(CONFIG_DIR, "european_country_mapping.csv")
BACI_PARQUET = os.path.join(DATA_DIR, "baci_hs92_seafood_1995_2024.parquet")

TOP20_PARTNERS = [
    ('CHN', 'China'), ('ECU', 'Ecuador'), ('MAR', 'Morocco'), ('VNM', 'Viet Nam'),
    ('USA', 'USA'), ('IND', 'India'), ('RUS', 'Russian Federation'), ('ARG', 'Argentina'),
    ('GRL', 'Greenland'), ('FRO', 'Faroe Islands'), ('CAN', 'Canada'), ('CHL', 'Chile'),
    ('THA', 'Thailand'), ('PER', 'Peru'), ('IDN', 'Indonesia'), ('NAM', 'Namibia'),
    ('MRT', 'Mauritania'), ('SYC', 'Seychelles'), ('BGD', 'Bangladesh'), ('MUS', 'Mauritius')
]

def run_mirror_check():
    print("=" * 80)
    print("STEP 11: BACI HARMONIZED MIRROR & CIF/FOB BENCHMARK (2015-2024)")
    print("=" * 80)

    top20_iso = [p[0] for p in TOP20_PARTNERS]
    name_map = dict(TOP20_PARTNERS)

    # 1. Load Comtrade European imports (CIF)
    print(f"[*] Querying Comtrade bilateral imports: {BILATERAL_PARQUET}")
    trade_dataset = ds.dataset(BILATERAL_PARQUET, format="parquet")
    c_table = trade_dataset.to_table(
        filter=(ds.field('Year') >= 2015) & (ds.field('Year') <= 2024) &
               (ds.field('Trade_Flow') == 'Import') & (ds.field('Partner_ISO3').isin(top20_iso)),
        columns=['Partner_ISO3', 'Trade_Value_USD', 'Weight_KG']
    )
    df_c = c_table.to_pandas()
    df_c['Partner_ISO3'] = df_c['Partner_ISO3'].str.strip().str.upper()

    agg_c = df_c.groupby('Partner_ISO3').agg(
        Comtrade_CIF_USD_B=('Trade_Value_USD', lambda x: x.sum() / 1e9),
        Comtrade_Volume_kt=('Weight_KG', lambda x: x.sum() / 1e6)
    ).reset_index()

    # 2. Load BACI reconciled flows (FOB)
    print(f"[*] Querying CEPII BACI global database: {BACI_PARQUET}")
    df_map = pd.read_csv(COUNTRY_MAP_CSV)
    focal_iso = set(df_map[df_map['Is_Focal_European']]['ISO3'].str.strip().str.upper().unique())

    b_dataset = ds.dataset(BACI_PARQUET, format="parquet")
    b_table = b_dataset.to_table(
        filter=(ds.field('Year') >= 2015) & (ds.field('Year') <= 2024) &
               (ds.field('Exporter_ISO3').isin(top20_iso)) & (ds.field('Importer_ISO3').isin(focal_iso)),
        columns=['Exporter_ISO3', 'Value_USD', 'Quantity_t']
    )
    df_b = b_table.to_pandas()
    df_b['Exporter_ISO3'] = df_b['Exporter_ISO3'].str.strip().str.upper()

    agg_b = df_b.groupby('Exporter_ISO3').agg(
        BACI_FOB_USD_B=('Value_USD', lambda x: x.sum() / 1e9),
        BACI_Volume_kt=('Quantity_t', lambda x: x.sum() / 1e3)
    ).reset_index().rename(columns={'Exporter_ISO3': 'Partner_ISO3'})

    # Merge
    merged = pd.merge(agg_c, agg_b, on='Partner_ISO3', how='left')
    merged['Country_Name'] = merged['Partner_ISO3'].map(name_map)

    # For FRO: BACI aggregates Faroe Islands under Denmark; document explicitly
    merged['BACI_FOB_USD_B'] = merged['BACI_FOB_USD_B'].fillna(np.nan)
    merged['BACI_Volume_kt'] = merged['BACI_Volume_kt'].fillna(np.nan)

    # CIF-FOB Freight Margin: (Comtrade CIF - BACI FOB) / Comtrade CIF
    # When BACI FOB is based on exporter declarations adjusted for world average freight,
    # ratio reflects ocean shipping markup and mirror discrepancy.
    merged['CIF_FOB_Value_Wedge_Pct'] = np.where(
        merged['BACI_FOB_USD_B'].notnull() & (merged['Comtrade_CIF_USD_B'] > 0),
        ((merged['Comtrade_CIF_USD_B'] - merged['BACI_FOB_USD_B']) / merged['Comtrade_CIF_USD_B']) * 100.0,
        np.nan
    )
    merged['Physical_Volume_Ratio_Comtrade_to_BACI'] = np.where(
        merged['BACI_Volume_kt'] > 0,
        merged['Comtrade_Volume_kt'] / merged['BACI_Volume_kt'],
        np.nan
    )

    # Order columns
    cols_order = [
        'Partner_ISO3', 'Country_Name', 'Comtrade_CIF_USD_B', 'BACI_FOB_USD_B',
        'CIF_FOB_Value_Wedge_Pct', 'Comtrade_Volume_kt', 'BACI_Volume_kt',
        'Physical_Volume_Ratio_Comtrade_to_BACI'
    ]
    merged = merged.sort_values('Comtrade_CIF_USD_B', ascending=False).reset_index(drop=True)
    table_mirror = merged[cols_order]

    out_path = os.path.join(TABLES_DIR, "Table_SI_BACI_Mirror_Check_Top20.csv")
    table_mirror.round(2).to_csv(out_path, index=False)
    print(f"[+] Saved BACI Mirror Check Table to: {out_path}")

    # Summary
    valid = merged.dropna(subset=['BACI_FOB_USD_B'])
    c_tot = valid['Comtrade_CIF_USD_B'].sum()
    b_tot = valid['BACI_FOB_USD_B'].sum()
    cw_tot = valid['Comtrade_Volume_kt'].sum()
    bw_tot = valid['BACI_Volume_kt'].sum()

    print(f"\n[+] BACI Mirror Robustness Summary (19 Reconciled Corridors):")
    print(f"    - Comtrade Total CIF: ${c_tot:.2f}B | Volume: {cw_tot:.1f} kt")
    print(f"    - BACI Total FOB:     ${b_tot:.2f}B | Volume: {bw_tot:.1f} kt")
    print(f"    - Aggregate Physical Biomass Consistency: {cw_tot/bw_tot*100:.1f}%")
    print(f"    - Mean Corridor Physical Alignment: {valid['Physical_Volume_Ratio_Comtrade_to_BACI'].mean():.2f}")
    print("=" * 80)
    print("STEP 11 COMPLETED SUCCESSFULLY.")
    print("=" * 80)

if __name__ == "__main__":
    run_mirror_check()
