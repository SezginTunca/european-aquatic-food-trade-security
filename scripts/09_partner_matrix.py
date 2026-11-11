#!/usr/bin/env python3
"""
scripts/09_partner_matrix.py
============================
Disaggregates European extra-continental seafood imports into an empirical
Top-20 Third-Country Partner Sourcing Matrix and 6 Supplier Archetypes (2015-2024).

Generates:
1. Table_Top20_Partner_Sourcing_Matrix.csv:
   Partner ISO3, Country Name, Supplier Archetype, Import Value ($B), Value Share (%),
   Import Volume (kt), Volume Share (%), CIF Unit Value ($/kg), Dominant Species Group,
   Dominant Processing Form, Production Mode, Primary Risk Class.
2. Table_Supplier_Archetypes_Summary.csv:
   Supplier Archetype, Partner Economies, Total Value ($B), Value Share (%),
   Total Volume (kt), Volume Share (%), Mean CIF Unit Value ($/kg), Dominant Species,
   Dominant Processing Form.
3. Table_Regional_Partner_HHI.csv:
   Consumer Macro-Region, Import Value ($B), Import Volume (kt), Extra-EU Value Share (%),
   Extra-EU Partner Sourcing HHI, All-Partner Sourcing HHI, Top-3 Third-Country Suppliers & Shares.
4. Table_Archetype_Unit_Values_by_Region.csv:
   Mean CIF unit values ($/kg) across supplier archetypes and consumer macro-regions.
"""

import os
import sys
import numpy as np
import pandas as pd
import pyarrow.dataset as ds

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "derived")
TABLES_DIR = os.path.join(PROJECT_ROOT, "tables")
CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")
os.makedirs(TABLES_DIR, exist_ok=True)

BILATERAL_PARQUET = os.path.join(DATA_DIR, "eu_species_bilateral_panel.parquet")
COUNTRY_MAP_CSV = os.path.join(CONFIG_DIR, "european_country_mapping.csv")
REJECTIONS_CSV = os.path.join(DATA_DIR, "eu_fao_border_rejections.csv")

# 6 Canonical Supplier Archetypes
SUPPLIER_ARCHETYPES = {
    'Tropical Farmed Aquaculture': ['ECU', 'VNM', 'IND', 'IDN', 'BGD'],
    'North Atlantic & Arctic Capture': ['RUS', 'GRL'],
    'Cold-Water Farmed Salmonids': ['FRO', 'CHL'],
    'Upwelling & Ocean Tuna Fisheries': ['MAR', 'PER', 'NAM', 'MRT', 'SYC', 'MUS'],
    'Global Processing Hubs': ['CHN', 'THA'],
    'Americas Marine Capture': ['USA', 'ARG', 'CAN']
}

ARCHETYPE_LOOKUP = {}
for arch, isos in SUPPLIER_ARCHETYPES.items():
    for iso in isos:
        ARCHETYPE_LOOKUP[iso] = arch

# Macro-regions
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

def run_partner_matrix():
    print("=" * 80)
    print("STEP 09: TOP-20 THIRD-COUNTRY PARTNER MATRIX & ARCHETYPES (2015-2024)")
    print("=" * 80)

    # 1. Load Bilateral Dataset (2015-2024 Imports)
    print(f"[*] Querying bilateral trade panel: {BILATERAL_PARQUET}")
    trade_dataset = ds.dataset(BILATERAL_PARQUET, format="parquet")
    trade_table = trade_dataset.to_table(
        filter=(ds.field('Year') >= 2015) & (ds.field('Year') <= 2024) & (ds.field('Trade_Flow') == 'Import'),
        columns=[
            'Year', 'Reporter_ISO3', 'Partner_ISO3', 'Partner_Country',
            'Species_Group', 'Processing_Form', 'Trade_Value_USD', 'Weight_KG',
            'Is_Extra_European'
        ]
    )
    df = trade_table.to_pandas()
    df['Partner_ISO3'] = df['Partner_ISO3'].str.strip().str.upper()
    df['Reporter_ISO3'] = df['Reporter_ISO3'].str.strip().str.upper()
    df['Weight_Tonnes'] = df['Weight_KG'] / 1e3

    total_import_val = df['Trade_Value_USD'].sum()
    total_import_wt = df['Weight_Tonnes'].sum()

    extra_df = df[df['Is_Extra_European']].copy()
    total_extra_val = extra_df['Trade_Value_USD'].sum()
    total_extra_wt = extra_df['Weight_Tonnes'].sum()

    print(f"    Total European Imports (2015-2024): ${total_import_val/1e9:.2f}B | {total_import_wt/1e6:.2f} MMT")
    print(f"    Extra-European Imports (2015-2024): ${total_extra_val/1e9:.2f}B ({total_extra_val/total_import_val*100:.1f}%) | "
          f"{total_extra_wt/1e6:.2f} MMT ({total_extra_wt/total_import_wt*100:.1f}%)")

    # 2. Identify Top 20 Extra-European Partners
    partner_agg = extra_df.groupby(['Partner_ISO3', 'Partner_Country']).agg(
        Trade_Value_USD=('Trade_Value_USD', 'sum'),
        Weight_Tonnes=('Weight_Tonnes', 'sum')
    ).reset_index()

    partner_agg['CIF_Unit_Value_USD_kg'] = partner_agg['Trade_Value_USD'] / (partner_agg['Weight_Tonnes'] * 1000.0)
    partner_agg['Value_Share_Extra_Pct'] = (partner_agg['Trade_Value_USD'] / total_extra_val) * 100.0
    partner_agg['Volume_Share_Extra_Pct'] = (partner_agg['Weight_Tonnes'] / total_extra_wt) * 100.0

    top20 = partner_agg.sort_values('Trade_Value_USD', ascending=False).head(20).copy().reset_index(drop=True)
    top20_iso_set = set(top20['Partner_ISO3'].unique())

    cum_top20_val = top20['Trade_Value_USD'].sum()
    cum_top20_wt = top20['Weight_Tonnes'].sum()
    print(f"    Top 20 Extra-European Partners: ${cum_top20_val/1e9:.2f}B ({cum_top20_val/total_extra_val*100:.1f}% of Extra-EU) | "
          f"{cum_top20_wt/1e6:.2f} MMT ({cum_top20_wt/total_extra_wt*100:.1f}% of Extra-EU)")

    # Dominant species and processing forms for Top 20
    dom_species = extra_df[extra_df['Partner_ISO3'].isin(top20_iso_set)].groupby(['Partner_ISO3', 'Species_Group'])['Trade_Value_USD'].sum().reset_index()
    top_species_idx = dom_species.groupby('Partner_ISO3')['Trade_Value_USD'].idxmax()
    partner_dom_species = dom_species.loc[top_species_idx].rename(columns={'Species_Group': 'Dominant_Species_Group'}).drop(columns=['Trade_Value_USD'])

    dom_form = extra_df[extra_df['Partner_ISO3'].isin(top20_iso_set)].groupby(['Partner_ISO3', 'Processing_Form'])['Trade_Value_USD'].sum().reset_index()
    top_form_idx = dom_form.groupby('Partner_ISO3')['Trade_Value_USD'].idxmax()
    partner_dom_form = dom_form.loc[top_form_idx].rename(columns={'Processing_Form': 'Dominant_Processing_Form'}).drop(columns=['Trade_Value_USD'])

    # Production Mode and Risk Class annotations
    prod_mode_map = {
        'CHN': 'Reprocessed Wild Marine & Farmed',
        'ECU': 'Intensive Coastal Aquaculture',
        'MAR': 'Upwelling Marine Wild Capture',
        'VNM': 'Intensive Freshwater & Brackish Aquaculture',
        'USA': 'Marine Wild Capture',
        'IND': 'Intensive Coastal Aquaculture',
        'RUS': 'Arctic & Boreal Marine Wild Capture',
        'ARG': 'South Atlantic Marine Wild Capture',
        'GRL': 'Arctic Marine Wild Capture',
        'FRO': 'Marine Aquaculture & Pelagic Capture',
        'CAN': 'Marine Wild Capture',
        'CHL': 'Marine Cold-Water Aquaculture',
        'THA': 'Reprocessed Tuna & Processed Shrimp',
        'PER': 'Upwelling Marine Wild Capture',
        'IDN': 'Coastal Aquaculture & Tuna Capture',
        'NAM': 'Benguela Upwelling Marine Capture',
        'MRT': 'Canary Current Marine Capture',
        'SYC': 'Western Indian Ocean Tuna Capture',
        'BGD': 'Inland Freshwater Aquaculture',
        'MUS': 'Western Indian Ocean Tuna Capture'
    }

    # Dominant Hazard Type from RASFF
    df_rej = pd.read_csv(REJECTIONS_CSV)
    df_rej['Exporting_ISO3'] = df_rej['Exporting_Country_ISO3'].str.strip().str.upper()
    top_hazards = df_rej[df_rej['Exporting_ISO3'].isin(top20_iso_set)].groupby(['Exporting_ISO3', 'Product_Category'])['Rejections_Count'].sum().reset_index()
    rej_haz_grp = df_rej[df_rej['Exporting_ISO3'].isin(top20_iso_set)].groupby(['Exporting_ISO3', 'Rejection_Cause'])['Rejections_Count'].sum().reset_index()
    top_haz_idx = rej_haz_grp.groupby('Exporting_ISO3')['Rejections_Count'].idxmax()
    dom_haz = rej_haz_grp.loc[top_haz_idx].rename(columns={'Exporting_ISO3': 'Partner_ISO3', 'Rejection_Cause': 'Dominant_Rejection_Cause'}).drop(columns=['Rejections_Count'])

    top20 = pd.merge(top20, partner_dom_species, on='Partner_ISO3', how='left')
    top20 = pd.merge(top20, partner_dom_form, on='Partner_ISO3', how='left')
    top20 = pd.merge(top20, dom_haz, on='Partner_ISO3', how='left')
    top20['Supplier_Archetype'] = top20['Partner_ISO3'].map(ARCHETYPE_LOOKUP).fillna('Other Third Countries')
    top20['Production_Mode'] = top20['Partner_ISO3'].map(prod_mode_map).fillna('Mixed')
    top20['Dominant_Rejection_Cause'] = top20['Dominant_Rejection_Cause'].fillna('None / Low Frequency')

    # Format Table Top20
    top20_table = pd.DataFrame({
        'Rank': range(1, len(top20) + 1),
        'Partner_ISO3': top20['Partner_ISO3'],
        'Country_Name': top20['Partner_Country'],
        'Supplier_Archetype': top20['Supplier_Archetype'],
        'Import_Value_USD_B': (top20['Trade_Value_USD'] / 1e9).round(2),
        'Value_Share_Pct': top20['Value_Share_Extra_Pct'].round(2),
        'Import_Volume_kt': (top20['Weight_Tonnes'] / 1e3).round(1),
        'Volume_Share_Pct': top20['Volume_Share_Extra_Pct'].round(2),
        'CIF_Unit_Value_USD_kg': top20['CIF_Unit_Value_USD_kg'].round(2),
        'Dominant_Species_Group': top20['Dominant_Species_Group'],
        'Dominant_Processing_Form': top20['Dominant_Processing_Form'],
        'Production_Mode': top20['Production_Mode'],
        'Primary_Hazard_Vector': top20['Dominant_Rejection_Cause']
    })

    top20_path = os.path.join(TABLES_DIR, "Table_Top20_Partner_Sourcing_Matrix.csv")
    top20_table.to_csv(top20_path, index=False)
    print(f"[+] Saved Table Top20 Partner Sourcing Matrix to: {top20_path}")

    # 3. Supplier Archetypes Summary Table
    extra_df['Supplier_Archetype'] = extra_df['Partner_ISO3'].map(ARCHETYPE_LOOKUP).fillna('Other Third Countries')
    arch_summary = extra_df.groupby('Supplier_Archetype').agg(
        Total_Value_USD=('Trade_Value_USD', 'sum'),
        Total_Weight_Tonnes=('Weight_Tonnes', 'sum')
    ).reset_index()

    arch_summary['Value_Share_Pct'] = (arch_summary['Total_Value_USD'] / total_extra_val) * 100.0
    arch_summary['Volume_Share_Pct'] = (arch_summary['Total_Weight_Tonnes'] / total_extra_wt) * 100.0
    arch_summary['CIF_Unit_Value_USD_kg'] = arch_summary['Total_Value_USD'] / (arch_summary['Total_Weight_Tonnes'] * 1000.0)

    # Dominant species and form per archetype
    arch_spec = extra_df.groupby(['Supplier_Archetype', 'Species_Group'])['Trade_Value_USD'].sum().reset_index()
    top_arch_spec_idx = arch_spec.groupby('Supplier_Archetype')['Trade_Value_USD'].idxmax()
    arch_dom_species = arch_spec.loc[top_arch_spec_idx].rename(columns={'Species_Group': 'Dominant_Species_Group'}).drop(columns=['Trade_Value_USD'])

    arch_form = extra_df.groupby(['Supplier_Archetype', 'Processing_Form'])['Trade_Value_USD'].sum().reset_index()
    top_arch_form_idx = arch_form.groupby('Supplier_Archetype')['Trade_Value_USD'].idxmax()
    arch_dom_form = arch_form.loc[top_arch_form_idx].rename(columns={'Processing_Form': 'Dominant_Processing_Form'}).drop(columns=['Trade_Value_USD'])

    arch_summary = pd.merge(arch_summary, arch_dom_species, on='Supplier_Archetype', how='left')
    arch_summary = pd.merge(arch_summary, arch_dom_form, on='Supplier_Archetype', how='left')

    partners_per_arch = {k: ', '.join(v) for k, v in SUPPLIER_ARCHETYPES.items()}
    partners_per_arch['Other Third Countries'] = 'Rest of World'
    arch_summary['Partner_Economies'] = arch_summary['Supplier_Archetype'].map(partners_per_arch)

    arch_table = pd.DataFrame({
        'Supplier_Archetype': arch_summary['Supplier_Archetype'],
        'Partner_Economies': arch_summary['Partner_Economies'],
        'Total_Value_USD_B': (arch_summary['Total_Value_USD'] / 1e9).round(2),
        'Value_Share_Pct': arch_summary['Value_Share_Pct'].round(2),
        'Total_Volume_kt': (arch_summary['Total_Weight_Tonnes'] / 1e3).round(1),
        'Volume_Share_Pct': arch_summary['Volume_Share_Pct'].round(2),
        'Mean_CIF_Unit_Value_USD_kg': arch_summary['CIF_Unit_Value_USD_kg'].round(2),
        'Dominant_Species': arch_summary['Dominant_Species_Group'],
        'Dominant_Processing_Form': arch_summary['Dominant_Processing_Form']
    }).sort_values('Total_Value_USD_B', ascending=False)

    arch_path = os.path.join(TABLES_DIR, "Table_Supplier_Archetypes_Summary.csv")
    arch_table.to_csv(arch_path, index=False)
    print(f"[+] Saved Table Supplier Archetypes Summary to: {arch_path}")

    # 4. Partner HHI of Supply by European Macro-Region
    print("[*] Calculating Regional Partner Sourcing HHI...")
    df['Consumer_Macro_Region'] = df['Reporter_ISO3'].apply(get_macro_region)

    reg_records = []
    for reg, grp in df.groupby('Consumer_Macro_Region'):
        if reg == 'Other Europe':
            continue
        tot_reg_val = grp['Trade_Value_USD'].sum()
        tot_reg_wt = grp['Weight_Tonnes'].sum()

        # Extra-EU subset
        extra_grp = grp[grp['Is_Extra_European']].copy()
        extra_reg_val = extra_grp['Trade_Value_USD'].sum()
        extra_share_val = (extra_reg_val / tot_reg_val) * 100.0 if tot_reg_val > 0 else 0

        # Partner shares in extra-EU imports
        part_extra = extra_grp.groupby('Partner_ISO3')['Trade_Value_USD'].sum()
        shares_extra = part_extra / extra_reg_val
        hhi_extra = (shares_extra ** 2).sum() * 10000.0

        # Partner shares in all imports
        part_all = grp.groupby('Partner_ISO3')['Trade_Value_USD'].sum()
        shares_all = part_all / tot_reg_val
        hhi_all = (shares_all ** 2).sum() * 10000.0

        # Top 3 Extra-EU suppliers
        top3 = shares_extra.sort_values(ascending=False).head(3)
        top3_str = "; ".join([f"{p} ({s*100:.1f}%)" for p, s in top3.items()])

        reg_records.append({
            'Consumer_Macro_Region': reg,
            'Total_Import_Value_USD_B': round(tot_reg_val / 1e9, 2),
            'Total_Import_Volume_kt': round(tot_reg_wt / 1e3, 1),
            'Extra_EU_Value_Share_Pct': round(extra_share_val, 2),
            'Extra_EU_Sourcing_HHI': round(hhi_extra, 1),
            'All_Partner_Sourcing_HHI': round(hhi_all, 1),
            'Top_3_Third_Country_Suppliers': top3_str
        })

    df_reg_hhi = pd.DataFrame(reg_records).sort_values('Total_Import_Value_USD_B', ascending=False)
    reg_hhi_path = os.path.join(TABLES_DIR, "Table_Regional_Partner_HHI.csv")
    df_reg_hhi.to_csv(reg_hhi_path, index=False)
    print(f"[+] Saved Table Regional Partner HHI to: {reg_hhi_path}")

    # 5. Unit Values by Archetype and Region
    def map_full_archetype(row):
        if not row['Is_Extra_European']:
            return 'Intra-European Core'
        p = str(row['Partner_ISO3']).strip().upper()
        return ARCHETYPE_LOOKUP.get(p, 'Other Third Countries')

    df['Supplier_Archetype'] = df.apply(map_full_archetype, axis=1)
    arch_reg = df.groupby(['Supplier_Archetype', 'Consumer_Macro_Region']).agg(
        Val=('Trade_Value_USD', 'sum'),
        Wt=('Weight_Tonnes', 'sum')
    ).reset_index()
    arch_reg['Unit_Value_USD_kg'] = arch_reg['Val'] / (arch_reg['Wt'] * 1000.0)

    piv_arch_uv = arch_reg.pivot_table(
        index='Supplier_Archetype',
        columns='Consumer_Macro_Region',
        values='Unit_Value_USD_kg'
    ).reset_index()

    cols_reg = ['Supplier_Archetype', 'Western Europe', 'Southern Europe', 'Central & Eastern Europe', 'Northern Europe']
    piv_arch_uv = piv_arch_uv[[c for c in cols_reg if c in piv_arch_uv.columns]]
    arch_uv_path = os.path.join(TABLES_DIR, "Table_Archetype_Unit_Values_by_Region.csv")
    piv_arch_uv.round(2).to_csv(arch_uv_path, index=False)
    print(f"[+] Saved Table Archetype Unit Values by Region to: {arch_uv_path}")

    print("=" * 80)
    print("STEP 09 COMPLETED SUCCESSFULLY.")
    print("=" * 80)

if __name__ == "__main__":
    run_partner_matrix()
