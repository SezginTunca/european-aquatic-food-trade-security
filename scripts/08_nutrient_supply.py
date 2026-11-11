#!/usr/bin/env python3
"""
scripts/08_nutrient_supply.py
=============================
Nutrient-weighted aquatic food sourcing and micronutrient delivery models:

Analyzes:
1. Aquatic food composition mapping based on published coefficients from the
   Aquatic Foods Composition Database (Golden et al., 2021, Nature 598:315-320; FAO/INFOODS).
2. Estimates daily per-capita supply of:
   - Protein (g/capita/day)
   - EPA + DHA omega-3 long-chain fatty acids (mg/capita/day)
   - Vitamin B12 (ug/capita/day)
   - Iron & Zinc (mg/capita/day)
3. Traces nutrient supply across 32 European nations and 4 macro-regions (CEE, Western,
   Southern, Northern Europe) over 1990-2024.
4. Identifies nutritional trade specialization (e.g., pelagics and salmonids delivering
   omega-3s; bivalves delivering B12 and iron; whitefish delivering lean protein).

Outputs in tables/:
- Table_D_Nutrient_Composition_Coefficients.csv
- Table_D2_Nutrient_Weighted_Supply_Per_Capita.csv
"""

import os
import sys
import numpy as np
import pandas as pd
import pyarrow.dataset as ds

# Set deterministic seed
np.random.seed(42)

# Dynamic relative paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "derived")
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
TABLES_DIR = os.path.join(PROJECT_ROOT, "tables")
CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")
os.makedirs(TABLES_DIR, exist_ok=True)

CONSUMPTION_CSV = os.path.join(DATA_DIR, "eu_fao_consumption_self_sufficiency_1950_2024.csv")
POPULATION_CSV = os.path.join(RAW_DATA_DIR, "wdi_population_total_1960_2024.csv")
BILATERAL_PARQUET = os.path.join(DATA_DIR, "eu_species_bilateral_panel.parquet")
COUNTRY_MAP_CSV = os.path.join(CONFIG_DIR, "european_country_mapping.csv")

# ------------------------------------------------------------------------------
# 1. Aquatic Foods Composition Database (AFCD) Reference Values
# ------------------------------------------------------------------------------
# Published values per 100g raw edible portion, Golden et al. (2021) Nature
AFCD_COEFFICIENTS = [
    {
        'Species_Group': 'Atlantic & Pacific Salmon',
        'Edible_Yield_Fraction': 0.65,
        'Protein_g_per_100g': 20.4,
        'EPA_DHA_mg_per_100g': 1850.0,
        'Vitamin_B12_ug_per_100g': 3.20,
        'Iron_mg_per_100g': 0.40,
        'Zinc_mg_per_100g': 0.42,
        'Citation_Source': 'Golden et al. (2021) Nature 598:315-320 (AFCD); FAO/INFOODS uFiSh',
        'Data_Status': 'Approximate published reference value'
    },
    {
        'Species_Group': 'Bluefin Tuna & Major Tunas',
        'Edible_Yield_Fraction': 0.70,
        'Protein_g_per_100g': 23.3,
        'EPA_DHA_mg_per_100g': 850.0,
        'Vitamin_B12_ug_per_100g': 2.20,
        'Iron_mg_per_100g': 1.02,
        'Zinc_mg_per_100g': 0.60,
        'Citation_Source': 'Golden et al. (2021) Nature 598:315-320 (AFCD); FAO/INFOODS uFiSh',
        'Data_Status': 'Approximate published reference value'
    },
    {
        'Species_Group': 'Cod, Hake, Pollock & Whitefish',
        'Edible_Yield_Fraction': 0.55,
        'Protein_g_per_100g': 17.8,
        'EPA_DHA_mg_per_100g': 220.0,
        'Vitamin_B12_ug_per_100g': 1.10,
        'Iron_mg_per_100g': 0.32,
        'Zinc_mg_per_100g': 0.45,
        'Citation_Source': 'Golden et al. (2021) Nature 598:315-320 (AFCD); FAO/INFOODS uFiSh',
        'Data_Status': 'Approximate published reference value'
    },
    {
        'Species_Group': 'Crabs, Lobsters & Other Crustaceans',
        'Edible_Yield_Fraction': 0.45,
        'Protein_g_per_100g': 18.1,
        'EPA_DHA_mg_per_100g': 350.0,
        'Vitamin_B12_ug_per_100g': 4.00,
        'Iron_mg_per_100g': 1.25,
        'Zinc_mg_per_100g': 3.50,
        'Citation_Source': 'Golden et al. (2021) Nature 598:315-320 (AFCD); FAO/INFOODS uFiSh',
        'Data_Status': 'Approximate published reference value'
    },
    {
        'Species_Group': 'European Seabass',
        'Edible_Yield_Fraction': 0.60,
        'Protein_g_per_100g': 19.8,
        'EPA_DHA_mg_per_100g': 780.0,
        'Vitamin_B12_ug_per_100g': 2.00,
        'Iron_mg_per_100g': 0.38,
        'Zinc_mg_per_100g': 0.48,
        'Citation_Source': 'Golden et al. (2021) Nature 598:315-320 (AFCD); FAO/INFOODS uFiSh',
        'Data_Status': 'Approximate published reference value'
    },
    {
        'Species_Group': 'Flatfish & Turbot',
        'Edible_Yield_Fraction': 0.55,
        'Protein_g_per_100g': 18.2,
        'EPA_DHA_mg_per_100g': 410.0,
        'Vitamin_B12_ug_per_100g': 1.50,
        'Iron_mg_per_100g': 0.36,
        'Zinc_mg_per_100g': 0.52,
        'Citation_Source': 'Golden et al. (2021) Nature 598:315-320 (AFCD); FAO/INFOODS uFiSh',
        'Data_Status': 'Approximate published reference value'
    },
    {
        'Species_Group': 'Gilthead Seabream',
        'Edible_Yield_Fraction': 0.60,
        'Protein_g_per_100g': 19.7,
        'EPA_DHA_mg_per_100g': 720.0,
        'Vitamin_B12_ug_per_100g': 2.10,
        'Iron_mg_per_100g': 0.40,
        'Zinc_mg_per_100g': 0.46,
        'Citation_Source': 'Golden et al. (2021) Nature 598:315-320 (AFCD); FAO/INFOODS uFiSh',
        'Data_Status': 'Approximate published reference value'
    },
    {
        'Species_Group': 'Herring, Sardines & Small Pelagics',
        'Edible_Yield_Fraction': 0.70,
        'Protein_g_per_100g': 19.5,
        'EPA_DHA_mg_per_100g': 1980.0,
        'Vitamin_B12_ug_per_100g': 11.00,
        'Iron_mg_per_100g': 1.40,
        'Zinc_mg_per_100g': 1.30,
        'Citation_Source': 'Golden et al. (2021) Nature 598:315-320 (AFCD); FAO/INFOODS uFiSh',
        'Data_Status': 'Approximate published reference value'
    },
    {
        'Species_Group': 'Live Aquaculture Fry & Ornamental',
        'Edible_Yield_Fraction': 0.50,
        'Protein_g_per_100g': 16.0,
        'EPA_DHA_mg_per_100g': 300.0,
        'Vitamin_B12_ug_per_100g': 1.00,
        'Iron_mg_per_100g': 0.50,
        'Zinc_mg_per_100g': 0.50,
        'Citation_Source': 'Golden et al. (2021) Nature 598:315-320 (AFCD); FAO/INFOODS uFiSh',
        'Data_Status': 'Approximate published reference value'
    },
    {
        'Species_Group': 'Mackerel & Horse Mackerel',
        'Edible_Yield_Fraction': 0.65,
        'Protein_g_per_100g': 18.6,
        'EPA_DHA_mg_per_100g': 1650.0,
        'Vitamin_B12_ug_per_100g': 8.70,
        'Iron_mg_per_100g': 1.20,
        'Zinc_mg_per_100g': 0.88,
        'Citation_Source': 'Golden et al. (2021) Nature 598:315-320 (AFCD); FAO/INFOODS uFiSh',
        'Data_Status': 'Approximate published reference value'
    },
    {
        'Species_Group': 'Mussels, Oysters & Bivalves',
        'Edible_Yield_Fraction': 0.25,
        'Protein_g_per_100g': 11.9,
        'EPA_DHA_mg_per_100g': 480.0,
        'Vitamin_B12_ug_per_100g': 16.00,
        'Iron_mg_per_100g': 3.90,
        'Zinc_mg_per_100g': 2.70,
        'Citation_Source': 'Golden et al. (2021) Nature 598:315-320 (AFCD); FAO/INFOODS uFiSh',
        'Data_Status': 'Approximate published reference value'
    },
    {
        'Species_Group': 'Other Marine & Freshwater Finfish',
        'Edible_Yield_Fraction': 0.60,
        'Protein_g_per_100g': 18.5,
        'EPA_DHA_mg_per_100g': 500.0,
        'Vitamin_B12_ug_per_100g': 2.50,
        'Iron_mg_per_100g': 0.60,
        'Zinc_mg_per_100g': 0.60,
        'Citation_Source': 'Golden et al. (2021) Nature 598:315-320 (AFCD); FAO/INFOODS uFiSh',
        'Data_Status': 'Approximate published reference value'
    },
    {
        'Species_Group': 'Pangasius, Tilapia & Freshwater Catfish',
        'Edible_Yield_Fraction': 0.60,
        'Protein_g_per_100g': 16.2,
        'EPA_DHA_mg_per_100g': 180.0,
        'Vitamin_B12_ug_per_100g': 1.60,
        'Iron_mg_per_100g': 0.38,
        'Zinc_mg_per_100g': 0.50,
        'Citation_Source': 'Golden et al. (2021) Nature 598:315-320 (AFCD); FAO/INFOODS uFiSh',
        'Data_Status': 'Approximate published reference value'
    },
    {
        'Species_Group': 'Rainbow & Freshwater Trout',
        'Edible_Yield_Fraction': 0.65,
        'Protein_g_per_100g': 19.9,
        'EPA_DHA_mg_per_100g': 950.0,
        'Vitamin_B12_ug_per_100g': 4.50,
        'Iron_mg_per_100g': 0.48,
        'Zinc_mg_per_100g': 0.62,
        'Citation_Source': 'Golden et al. (2021) Nature 598:315-320 (AFCD); FAO/INFOODS uFiSh',
        'Data_Status': 'Approximate published reference value'
    },
    {
        'Species_Group': 'Sea Cucumber, Urchins & Invertebrates',
        'Edible_Yield_Fraction': 0.35,
        'Protein_g_per_100g': 13.0,
        'EPA_DHA_mg_per_100g': 250.0,
        'Vitamin_B12_ug_per_100g': 2.00,
        'Iron_mg_per_100g': 1.50,
        'Zinc_mg_per_100g': 1.80,
        'Citation_Source': 'Golden et al. (2021) Nature 598:315-320 (AFCD); FAO/INFOODS uFiSh',
        'Data_Status': 'Approximate published reference value'
    },
    {
        'Species_Group': 'Shrimp & Prawns',
        'Edible_Yield_Fraction': 0.55,
        'Protein_g_per_100g': 20.1,
        'EPA_DHA_mg_per_100g': 310.0,
        'Vitamin_B12_ug_per_100g': 1.50,
        'Iron_mg_per_100g': 0.52,
        'Zinc_mg_per_100g': 1.20,
        'Citation_Source': 'Golden et al. (2021) Nature 598:315-320 (AFCD); FAO/INFOODS uFiSh',
        'Data_Status': 'Approximate published reference value'
    },
    {
        'Species_Group': 'Squid, Cuttlefish & Octopus',
        'Edible_Yield_Fraction': 0.75,
        'Protein_g_per_100g': 16.5,
        'EPA_DHA_mg_per_100g': 490.0,
        'Vitamin_B12_ug_per_100g': 2.80,
        'Iron_mg_per_100g': 1.10,
        'Zinc_mg_per_100g': 1.50,
        'Citation_Source': 'Golden et al. (2021) Nature 598:315-320 (AFCD); FAO/INFOODS uFiSh',
        'Data_Status': 'Approximate published reference value'
    }
]

CEE_ISO = {'POL', 'CZE', 'SVK', 'HUN', 'ROU', 'BGR', 'LTU', 'LVA', 'EST', 'HRV', 'SVN'}
WEST_ISO = {'DEU', 'FRA', 'NLD', 'BEL', 'AUT', 'LUX', 'CHE', 'GBR', 'IRL'}
SOUTH_ISO = {'ESP', 'ITA', 'PRT', 'GRC', 'CYP', 'MLT', 'TUR'}
NORTH_ISO = {'NOR', 'SWE', 'FIN', 'DNK', 'ISL'}

def get_macro_region(iso: str) -> str:
    if iso in CEE_ISO:
        return 'Central & Eastern Europe'
    elif iso in WEST_ISO:
        return 'Western Europe'
    elif iso in SOUTH_ISO:
        return 'Southern Europe'
    elif iso in NORTH_ISO:
        return 'Northern Europe'
    return 'Other Europe'

def run_nutrient_module():
    print("=" * 80)
    print("STEP 08: NUTRIENT-WEIGHTED SUPPLY & MICRONUTRIENT ACCOUNTING")
    print("=" * 80)

    # 1. Save Table D: Nutrient Composition Coefficients
    df_afcd = pd.DataFrame(AFCD_COEFFICIENTS)
    table_d_path = os.path.join(TABLES_DIR, "Table_D_Nutrient_Composition_Coefficients.csv")
    df_afcd.to_csv(table_d_path, index=False)
    print(f"[+] Saved Table D (AFCD Reference Coefficients) to: {table_d_path}")

    # 2. Load consumption balances and population
    print(f"[*] Loading FAO consumption balances and population data...")
    df_cons = pd.read_csv(CONSUMPTION_CSV)
    df_pop = pd.read_csv(POPULATION_CSV)
    merged = pd.merge(df_cons, df_pop[['Country_ISO3', 'Year', 'Population']], on=['Country_ISO3', 'Year'], how='inner')
    merged['Per_Capita_Supply_KG'] = (merged['Apparent_Consumption_Tonnes'] * 1000.0) / merged['Population']
    merged['Macro_Region'] = merged['Country_ISO3'].apply(get_macro_region)

    # 3. Query trade panel for species composition by country and year
    print(f"[*] Querying species trade shares: {BILATERAL_PARQUET}")
    trade_dataset = ds.dataset(BILATERAL_PARQUET, format="parquet")
    trade_table = trade_dataset.to_table(
        filter=(ds.field('Year') >= 1990) & (ds.field('Trade_Flow') == 'Import'),
        columns=['Year', 'Reporter_ISO3', 'Species_Group', 'Weight_KG']
    ).to_pandas()

    sp_trade = trade_table.groupby(['Year', 'Reporter_ISO3', 'Species_Group'])['Weight_KG'].sum().reset_index()
    rep_tot = sp_trade.groupby(['Year', 'Reporter_ISO3'])['Weight_KG'].transform('sum')
    sp_trade['Species_Share'] = np.where(rep_tot > 0, sp_trade['Weight_KG'] / rep_tot, 1.0 / len(AFCD_COEFFICIENTS))

    # Merge with AFCD nutrient coefficients
    sp_trade = pd.merge(sp_trade, df_afcd, on='Species_Group', how='left')

    # Effective edible yield and nutrients per kg of product
    sp_trade['Edible_KG'] = sp_trade['Species_Share'] * sp_trade['Edible_Yield_Fraction']
    sp_trade['Protein_g_per_kg'] = sp_trade['Edible_KG'] * sp_trade['Protein_g_per_100g'] * 10.0
    sp_trade['EPA_DHA_mg_per_kg'] = sp_trade['Edible_KG'] * sp_trade['EPA_DHA_mg_per_100g'] * 10.0
    sp_trade['B12_ug_per_kg'] = sp_trade['Edible_KG'] * sp_trade['Vitamin_B12_ug_per_100g'] * 10.0
    sp_trade['Iron_mg_per_kg'] = sp_trade['Edible_KG'] * sp_trade['Iron_mg_per_100g'] * 10.0
    sp_trade['Zinc_mg_per_kg'] = sp_trade['Edible_KG'] * sp_trade['Zinc_mg_per_100g'] * 10.0

    # Aggregate nutrient factors per kg of seafood consumed for each country and year
    country_nutrient_factors = sp_trade.groupby(['Year', 'Reporter_ISO3']).agg({
        'Protein_g_per_kg': 'sum',
        'EPA_DHA_mg_per_kg': 'sum',
        'B12_ug_per_kg': 'sum',
        'Iron_mg_per_kg': 'sum',
        'Zinc_mg_per_kg': 'sum'
    }).reset_index().rename(columns={'Reporter_ISO3': 'Country_ISO3'})

    # Merge with per-capita apparent consumption
    df_nutrients = pd.merge(merged[merged['Year'] >= 1990], country_nutrient_factors, on=['Country_ISO3', 'Year'], how='left')

    # Fill default averages for any missing pairs
    default_prot = df_afcd['Protein_g_per_100g'].mean() * df_afcd['Edible_Yield_Fraction'].mean() * 10.0
    default_epa = df_afcd['EPA_DHA_mg_per_100g'].mean() * df_afcd['Edible_Yield_Fraction'].mean() * 10.0
    default_b12 = df_afcd['Vitamin_B12_ug_per_100g'].mean() * df_afcd['Edible_Yield_Fraction'].mean() * 10.0
    default_fe = df_afcd['Iron_mg_per_100g'].mean() * df_afcd['Edible_Yield_Fraction'].mean() * 10.0
    default_zn = df_afcd['Zinc_mg_per_100g'].mean() * df_afcd['Edible_Yield_Fraction'].mean() * 10.0

    df_nutrients['Protein_g_per_kg'] = df_nutrients['Protein_g_per_kg'].fillna(default_prot)
    df_nutrients['EPA_DHA_mg_per_kg'] = df_nutrients['EPA_DHA_mg_per_kg'].fillna(default_epa)
    df_nutrients['B12_ug_per_kg'] = df_nutrients['B12_ug_per_kg'].fillna(default_b12)
    df_nutrients['Iron_mg_per_kg'] = df_nutrients['Iron_mg_per_kg'].fillna(default_fe)
    df_nutrients['Zinc_mg_per_kg'] = df_nutrients['Zinc_mg_per_kg'].fillna(default_zn)

    # Compute daily per capita nutrient intakes
    # Daily = (Per_Capita_Supply_KG * nutrient_per_kg) / 365
    df_nutrients['Protein_g_per_day'] = (df_nutrients['Per_Capita_Supply_KG'] * df_nutrients['Protein_g_per_kg']) / 365.0
    df_nutrients['EPA_DHA_mg_per_day'] = (df_nutrients['Per_Capita_Supply_KG'] * df_nutrients['EPA_DHA_mg_per_kg']) / 365.0
    df_nutrients['Vitamin_B12_ug_per_day'] = (df_nutrients['Per_Capita_Supply_KG'] * df_nutrients['B12_ug_per_kg']) / 365.0
    df_nutrients['Iron_mg_per_day'] = (df_nutrients['Per_Capita_Supply_KG'] * df_nutrients['Iron_mg_per_kg']) / 365.0
    df_nutrients['Zinc_mg_per_day'] = (df_nutrients['Per_Capita_Supply_KG'] * df_nutrients['Zinc_mg_per_kg']) / 365.0

    table_d2_out = df_nutrients[df_nutrients['Year'].isin([1995, 2005, 2015, 2024])][[
        'Country_ISO3', 'Country_Name', 'Macro_Region', 'Year', 'Population',
        'Per_Capita_Supply_KG', 'Protein_g_per_day', 'EPA_DHA_mg_per_day',
        'Vitamin_B12_ug_per_day', 'Iron_mg_per_day', 'Zinc_mg_per_day'
    ]].sort_values(['Country_ISO3', 'Year'])

    table_d2_path = os.path.join(TABLES_DIR, "Table_D2_Nutrient_Weighted_Supply_Per_Capita.csv")
    table_d2_out.round(3).to_csv(table_d2_path, index=False)
    print(f"[+] Saved Table D2 to: {table_d2_path}")

    # Regional averages in 2024
    reg_2024 = df_nutrients[df_nutrients['Year'] == 2024].groupby('Macro_Region').agg({
        'Per_Capita_Supply_KG': 'mean',
        'Protein_g_per_day': 'mean',
        'EPA_DHA_mg_per_day': 'mean',
        'Vitamin_B12_ug_per_day': 'mean'
    }).reset_index()

    print("\n[+] 2024 Regional Aquatic Nutrient Sourcing (per capita / day):")
    for _, row in reg_2024.iterrows():
        print(f"    - {row['Macro_Region']}: Supply = {row['Per_Capita_Supply_KG']:.1f} kg/yr | "
              f"Protein = {row['Protein_g_per_day']:.1f} g/d | "
              f"EPA+DHA = {row['EPA_DHA_mg_per_day']:.0f} mg/d | "
              f"Vit B12 = {row['Vitamin_B12_ug_per_day']:.2f} ug/d")

    print("=" * 80)
    print("STEP 08 COMPLETED SUCCESSFULLY.")
    print("=" * 80)

if __name__ == "__main__":
    run_nutrient_module()
