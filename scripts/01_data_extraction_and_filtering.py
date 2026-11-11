#!/usr/bin/env python3
"""
scripts/01_data_extraction_and_filtering.py
============================================
Extracts, harmonizes, and compiles the dedicated European Seafood Trade datasets:
- EU-27 Member States
- Norway (EEA Partner)
- Türkiye (EU Customs Union & Mediterranean Aquaculture Leader)

Sources:
- Global Species Parquet: 25_Global_Seafood_Trade_Comtrade_Species/data/derived/master_global_bilateral_panel.parquet
- Global Commodity Parquet: 25_Global_Seafood_Trade_Commodities/data/derived/master_global_bilateral_sitc_panel_1962_2025.parquet
- Country Mapping: config/european_country_mapping.csv
- Species Taxonomy: config/species_taxonomy_master.csv
- SITC Taxonomy: config/sitc_commodity_taxonomy.csv

Outputs in data/derived/:
1. eu_species_bilateral_panel.parquet (1988-2025 micro-panel for EU27+TUR+NOR reporters)
2. eu_species_intra_panel.parquet (Pure Intra-European bilateral transactions)
3. eu_commodity_sitc_panel_1962_2025.parquet (64-year longitudinal commodity panel)
4. eu_annual_country_macro_balance.csv (Annual country-level balances, volumes, unit values, NTR)
5. eu_annual_species_trade.csv (Species-level turnover and pricing dynamics)
6. eu_annual_processing_form_trade.csv (Processing form value-addition distribution)
7. eu_commodity_macro_panel_1962_2025.csv (Longitudinal macro trends by era)
8. eu_commodity_category_panel_1962_2025.csv (4-category longitudinal evolution)
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Directories
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "derived")
TABLES_DIR = os.path.join(PROJECT_ROOT, "tables")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

# External sources
SP_SOURCE_DIR = "/Users/sezgintunca/_Coding_local/25_Global_Seafood_Trade_Comtrade_Species/data/derived"
CM_SOURCE_DIR = "/Users/sezgintunca/_Coding_local/25_Global_Seafood_Trade_Commodities/data/derived"

SP_PARQUET = os.path.join(SP_SOURCE_DIR, "master_global_bilateral_panel.parquet")
CM_PARQUET = os.path.join(CM_SOURCE_DIR, "master_global_bilateral_sitc_panel_1962_2025.parquet")

def load_mappings():
    map_file = os.path.join(CONFIG_DIR, "european_country_mapping.csv")
    if not os.path.exists(map_file):
        raise FileNotFoundError(f"Mapping not found: {map_file}")
    df_map = pd.read_csv(map_file)
    df_map['ISO3'] = df_map['ISO3'].astype(str).str.strip().str.upper()
    
    eu27_iso = set(df_map[df_map['Is_EU27']]['ISO3'].unique())
    focal_iso = set(df_map[df_map['Is_Focal_European']]['ISO3'].unique())
    
    cluster_lookup = dict(zip(df_map['ISO3'], df_map['European_Cluster']))
    status_lookup = dict(zip(df_map['ISO3'], df_map['Institutional_Status']))
    name_lookup = dict(zip(df_map['ISO3'], df_map['Country_Name']))
    
    return eu27_iso, focal_iso, cluster_lookup, status_lookup, name_lookup

def process_species_data(eu27_iso, focal_iso, cluster_lookup, status_lookup, name_lookup):
    print("\n" + "=" * 80)
    print("PHASE 1: EXTRACTING & STRUCTURING SPECIES-LEVEL BILATERAL PANEL (1988-2025)")
    print("=" * 80)
    t0 = time.time()
    
    if not os.path.exists(SP_PARQUET):
        raise FileNotFoundError(f"Missing species source parquet: {SP_PARQUET}")
        
    print(f"[*] Reading global species parquet: {SP_PARQUET}...")
    table = pq.read_table(SP_PARQUET)
    df = table.to_pandas()
    print(f"    Global records loaded: {len(df):,}")
    
    # Filter for Focal European Reporters (Official reported trade of EU27 + TUR + NOR)
    df['Reporter_ISO3'] = df['Reporter_ISO3'].astype(str).str.strip().str.upper()
    df['Partner_ISO3'] = df['Partner_ISO3'].astype(str).str.strip().str.upper()
    
    rep_focal = df['Reporter_ISO3'].isin(focal_iso)
    part_focal = df['Partner_ISO3'].isin(focal_iso)
    
    # Master European panel: Reporter in Focal European
    df_eu = df[rep_focal].copy()
    print(f"    Records with Reporter in EU-27 + TUR + NOR: {len(df_eu):,}")
    
    # Enrich with European institutional tags
    df_eu['Reporter_Cluster'] = df_eu['Reporter_ISO3'].map(cluster_lookup).fillna('Other')
    df_eu['Partner_Cluster'] = df_eu['Partner_ISO3'].map(cluster_lookup).fillna(df_eu['Partner_Bloc'])
    
    df_eu['Reporter_Status'] = df_eu['Reporter_ISO3'].map(status_lookup).fillna('Third Country')
    df_eu['Partner_Status'] = df_eu['Partner_ISO3'].map(status_lookup).fillna('Third Country')
    
    # Specific corridor flags
    df_eu['Is_Intra_European'] = df_eu['Partner_ISO3'].isin(focal_iso)
    df_eu['Is_Intra_EU27'] = df_eu['Reporter_ISO3'].isin(eu27_iso) & df_eu['Partner_ISO3'].isin(eu27_iso)
    df_eu['Is_Norway_Corridor'] = (
        ((df_eu['Reporter_ISO3'] == 'NOR') & df_eu['Partner_ISO3'].isin(eu27_iso)) |
        (df_eu['Reporter_ISO3'].isin(eu27_iso) & (df_eu['Partner_ISO3'] == 'NOR'))
    )
    df_eu['Is_Turkey_Corridor'] = (
        ((df_eu['Reporter_ISO3'] == 'TUR') & df_eu['Partner_ISO3'].isin(eu27_iso)) |
        (df_eu['Reporter_ISO3'].isin(eu27_iso) & (df_eu['Partner_ISO3'] == 'TUR'))
    )
    df_eu['Is_Iceland_Corridor'] = (
        ((df_eu['Reporter_ISO3'] == 'ISL') & df_eu['Partner_ISO3'].isin(eu27_iso)) |
        (df_eu['Reporter_ISO3'].isin(eu27_iso) & (df_eu['Partner_ISO3'] == 'ISL'))
    )
    df_eu['Is_Switzerland_Corridor'] = (
        ((df_eu['Reporter_ISO3'] == 'CHE') & df_eu['Partner_ISO3'].isin(eu27_iso)) |
        (df_eu['Reporter_ISO3'].isin(eu27_iso) & (df_eu['Partner_ISO3'] == 'CHE'))
    )
    df_eu['Is_UK_Corridor'] = (
        ((df_eu['Reporter_ISO3'] == 'GBR') & df_eu['Partner_ISO3'].isin(eu27_iso)) |
        (df_eu['Reporter_ISO3'].isin(eu27_iso) & (df_eu['Partner_ISO3'] == 'GBR'))
    )
    df_eu['Is_Extra_European'] = ~df_eu['Is_Intra_European']
    
    # Save full EU species panel
    out_eu_parquet = os.path.join(DATA_DIR, "eu_species_bilateral_panel.parquet")
    print(f"[*] Writing {out_eu_parquet}...")
    df_eu.to_parquet(out_eu_parquet, index=False, compression="snappy")
    print(f"    File size: {os.path.getsize(out_eu_parquet) / (1024*1024):.2f} MB")
    
    # Save pure intra-European subset
    df_intra = df_eu[df_eu['Is_Intra_European']].copy()
    out_intra_parquet = os.path.join(DATA_DIR, "eu_species_intra_panel.parquet")
    print(f"[*] Writing pure intra-European panel: {out_intra_parquet} ({len(df_intra):,} rows)...")
    df_intra.to_parquet(out_intra_parquet, index=False, compression="snappy")
    print(f"    File size: {os.path.getsize(out_intra_parquet) / (1024*1024):.2f} MB")
    
    # Compute Country Annual Macro Balance
    print("[*] Computing annual country macro balances...")
    macro_records = []
    years = sorted(df_eu['Year'].unique())
    
    for yr in years:
        df_yr = df_eu[df_eu['Year'] == yr]
        for iso in sorted(focal_iso):
            df_iso = df_yr[df_yr['Reporter_ISO3'] == iso]
            if df_iso.empty:
                continue
            
            c_name = name_lookup.get(iso, df_iso['Reporter_Country'].iloc[0])
            cluster = cluster_lookup.get(iso, 'Other')
            status = status_lookup.get(iso, 'Other')
            
            exp_sub = df_iso[df_iso['Trade_Flow'] == 'Export']
            imp_sub = df_iso[df_iso['Trade_Flow'] == 'Import']
            
            x_val = exp_sub['Trade_Value_USD'].sum()
            m_val = imp_sub['Trade_Value_USD'].sum()
            x_wgt = exp_sub['Weight_KG'].sum()
            m_wgt = imp_sub['Weight_KG'].sum()
            
            # Intra-European subcomponents
            x_val_intra = exp_sub[exp_sub['Is_Intra_European']]['Trade_Value_USD'].sum()
            m_val_intra = imp_sub[imp_sub['Is_Intra_European']]['Trade_Value_USD'].sum()
            
            # Trophic levels
            mtl_x = np.average(exp_sub['Trophic_Level'], weights=exp_sub['Trade_Value_USD']) if (not exp_sub.empty and x_val > 0) else np.nan
            mtl_m = np.average(imp_sub['Trophic_Level'], weights=imp_sub['Trade_Value_USD']) if (not imp_sub.empty and m_val > 0) else np.nan
            
            # Unit values
            uv_x = (x_val / x_wgt) if x_wgt > 0 else np.nan
            uv_m = (m_val / m_wgt) if m_wgt > 0 else np.nan
            
            # Net metrics
            turnover = x_val + m_val
            net_bal = x_val - m_val
            ntr = (net_bal / turnover) if turnover > 0 else np.nan
            intra_share_x = (x_val_intra / x_val) if x_val > 0 else np.nan
            intra_share_m = (m_val_intra / m_val) if m_val > 0 else np.nan
            
            macro_records.append({
                'Year': yr,
                'Reporter_ISO3': iso,
                'Country_Name': c_name,
                'European_Cluster': cluster,
                'Institutional_Status': status,
                'Export_Value_USD': x_val,
                'Import_Value_USD': m_val,
                'Export_Weight_KG': x_wgt,
                'Import_Weight_KG': m_wgt,
                'Total_Turnover_USD': turnover,
                'Net_Trade_Balance_USD': net_bal,
                'Net_Trade_Ratio': ntr,
                'Export_Unit_Value_USD_KG': uv_x,
                'Import_Unit_Value_USD_KG': uv_m,
                'MTL_Export': mtl_x,
                'MTL_Import': mtl_m,
                'Delta_MTL': (mtl_x - mtl_m) if (not np.isnan(mtl_x) and not np.isnan(mtl_m)) else np.nan,
                'Intra_European_Export_Share': intra_share_x,
                'Intra_European_Import_Share': intra_share_m
            })
            
    df_macro = pd.DataFrame(macro_records)
    out_macro = os.path.join(DATA_DIR, "eu_annual_country_macro_balance.csv")
    df_macro.to_csv(out_macro, index=False)
    print(f"    Saved macro balances: {out_macro} ({len(df_macro)} rows)")
    
    # Annual Species-Level Turnover
    print("[*] Computing annual species-level dynamics...")
    sp_agg = df_eu.groupby(['Year', 'Species_Group', 'Trade_Flow']).agg(
        Total_Value_USD=('Trade_Value_USD', 'sum'),
        Total_Weight_KG=('Weight_KG', 'sum'),
        Active_Reporters=('Reporter_ISO3', 'nunique'),
        Active_Partners=('Partner_ISO3', 'nunique')
    ).reset_index()
    sp_agg['Unit_Price_USD_per_KG'] = np.where(sp_agg['Total_Weight_KG'] > 0, sp_agg['Total_Value_USD'] / sp_agg['Total_Weight_KG'], np.nan)
    out_sp = os.path.join(DATA_DIR, "eu_annual_species_trade.csv")
    sp_agg.to_csv(out_sp, index=False)
    print(f"    Saved annual species trade: {out_sp} ({len(sp_agg)} rows)")
    
    # Annual Processing Form Dynamics
    print("[*] Computing processing form dynamics...")
    proc_agg = df_eu.groupby(['Year', 'Processing_Form', 'Trade_Flow']).agg(
        Total_Value_USD=('Trade_Value_USD', 'sum'),
        Total_Weight_KG=('Weight_KG', 'sum')
    ).reset_index()
    proc_agg['Unit_Price_USD_per_KG'] = np.where(proc_agg['Total_Weight_KG'] > 0, proc_agg['Total_Value_USD'] / proc_agg['Total_Weight_KG'], np.nan)
    out_proc = os.path.join(DATA_DIR, "eu_annual_processing_form_trade.csv")
    proc_agg.to_csv(out_proc, index=False)
    print(f"    Saved processing form trade: {out_proc} ({len(proc_agg)} rows)")
    
    print(f"[✓] Phase 1 completed in {time.time() - t0:.2f} seconds.")

def process_commodity_data(eu27_iso, focal_iso, cluster_lookup, status_lookup, name_lookup):
    print("\n" + "=" * 80)
    print("PHASE 2: EXTRACTING & STRUCTURING 64-YEAR COMMODITY SITC PANEL (1962-2025)")
    print("=" * 80)
    t0 = time.time()
    
    if not os.path.exists(CM_PARQUET):
        raise FileNotFoundError(f"Missing commodity source parquet: {CM_PARQUET}")
        
    print(f"[*] Reading global commodity parquet: {CM_PARQUET}...")
    table = pq.read_table(CM_PARQUET)
    df = table.to_pandas()
    print(f"    Global commodity records: {len(df):,}")
    
    df['Reporter_ISO3'] = df['Reporter_ISO3'].astype(str).str.strip().str.upper()
    df['Partner_ISO3'] = df['Partner_ISO3'].astype(str).str.strip().str.upper()
    
    rep_focal = df['Reporter_ISO3'].isin(focal_iso)
    df_eu_cm = df[rep_focal].copy()
    print(f"    Records with Reporter in EU-27 + TUR + NOR: {len(df_eu_cm):,}")
    
    df_eu_cm['Reporter_Cluster'] = df_eu_cm['Reporter_ISO3'].map(cluster_lookup).fillna('Other')
    df_eu_cm['Partner_Cluster'] = df_eu_cm['Partner_ISO3'].map(cluster_lookup).fillna(df_eu_cm['Partner_Bloc'])
    df_eu_cm['Reporter_Status'] = df_eu_cm['Reporter_ISO3'].map(status_lookup).fillna('Third Country')
    df_eu_cm['Partner_Status'] = df_eu_cm['Partner_ISO3'].map(status_lookup).fillna('Third Country')
    
    df_eu_cm['Is_Intra_European'] = df_eu_cm['Partner_ISO3'].isin(focal_iso)
    df_eu_cm['Is_Intra_EU27'] = df_eu_cm['Reporter_ISO3'].isin(eu27_iso) & df_eu_cm['Partner_ISO3'].isin(eu27_iso)
    
    out_cm_parquet = os.path.join(DATA_DIR, "eu_commodity_sitc_panel_1962_2025.parquet")
    print(f"[*] Writing {out_cm_parquet}...")
    df_eu_cm.to_parquet(out_cm_parquet, index=False, compression="snappy")
    print(f"    File size: {os.path.getsize(out_cm_parquet) / (1024*1024):.2f} MB")
    
    # 64-Year Macro Aggregation
    print("[*] Computing 64-year longitudinal macro time series...")
    macro_cm = df_eu_cm.groupby(['Year', 'Era', 'Era_Label', 'Trade_Flow']).agg(
        Total_Value_USD=('Trade_Value_USD', 'sum'),
        Total_Weight_KG=('Weight_KG', 'sum'),
        Active_Reporters=('Reporter_ISO3', 'nunique'),
        Active_Partners=('Partner_ISO3', 'nunique')
    ).reset_index()
    
    macro_cm_pivot = macro_cm.pivot_table(
        index=['Year', 'Era', 'Era_Label'],
        columns='Trade_Flow',
        values=['Total_Value_USD', 'Total_Weight_KG'],
        aggfunc='sum'
    ).reset_index()
    
    macro_cm_pivot.columns = [
        '_'.join(col).strip('_') if col[1] else col[0] for col in macro_cm_pivot.columns
    ]
    
    # Clean column names
    x_val_col = 'Total_Value_USD_Export' if 'Total_Value_USD_Export' in macro_cm_pivot.columns else None
    m_val_col = 'Total_Value_USD_Import' if 'Total_Value_USD_Import' in macro_cm_pivot.columns else None
    x_wgt_col = 'Total_Weight_KG_Export' if 'Total_Weight_KG_Export' in macro_cm_pivot.columns else None
    m_wgt_col = 'Total_Weight_KG_Import' if 'Total_Weight_KG_Import' in macro_cm_pivot.columns else None
    
    if x_val_col and m_val_col:
        macro_cm_pivot['Total_Turnover_USD_B'] = (macro_cm_pivot[x_val_col] + macro_cm_pivot[m_val_col]) / 1e9
        macro_cm_pivot['Net_Balance_USD_B'] = (macro_cm_pivot[x_val_col] - macro_cm_pivot[m_val_col]) / 1e9
        macro_cm_pivot['Export_USD_B'] = macro_cm_pivot[x_val_col] / 1e9
        macro_cm_pivot['Import_USD_B'] = macro_cm_pivot[m_val_col] / 1e9
        macro_cm_pivot['Grubel_Lloyd_IIT'] = 1 - (
            abs(macro_cm_pivot[x_val_col] - macro_cm_pivot[m_val_col]) /
            (macro_cm_pivot[x_val_col] + macro_cm_pivot[m_val_col])
        )
    if x_wgt_col and m_wgt_col:
        macro_cm_pivot['Export_Weight_MMT'] = macro_cm_pivot[x_wgt_col] / 1e9
        macro_cm_pivot['Import_Weight_MMT'] = macro_cm_pivot[m_wgt_col] / 1e9
        
    out_cm_macro = os.path.join(DATA_DIR, "eu_commodity_macro_panel_1962_2025.csv")
    macro_cm_pivot.to_csv(out_cm_macro, index=False)
    print(f"    Saved commodity macro series: {out_cm_macro} ({len(macro_cm_pivot)} rows)")
    
    # 4-Category Longitudinal Breakdown
    print("[*] Computing 4-category longitudinal evolution...")
    cat_cm = df_eu_cm.groupby(['Year', 'Era', 'Era_Label', 'Harmonized_Category', 'Trade_Flow']).agg(
        Trade_Value_USD=('Trade_Value_USD', 'sum'),
        Weight_KG=('Weight_KG', 'sum')
    ).reset_index()
    cat_cm['Unit_Price_USD_per_KG'] = np.where(cat_cm['Weight_KG'] > 0, cat_cm['Trade_Value_USD'] / cat_cm['Weight_KG'], np.nan)
    cat_cm['Trade_Value_USD_M'] = cat_cm['Trade_Value_USD'] / 1e6
    cat_cm['Weight_MT'] = cat_cm['Weight_KG'] / 1e3
    
    out_cm_cat = os.path.join(DATA_DIR, "eu_commodity_category_panel_1962_2025.csv")
    cat_cm.to_csv(out_cm_cat, index=False)
    print(f"    Saved commodity category panel: {out_cm_cat} ({len(cat_cm)} rows)")
    
    print(f"[✓] Phase 2 completed in {time.time() - t0:.2f} seconds.")

def process_fao_consumption_data(focal_iso, name_lookup):
    print("\n" + "=" * 80)
    print("PHASE 3: EXTRACTING FAO CONSUMPTION, PRODUCTION, EMPLOYMENT & BORDER REJECTIONS")
    print("=" * 80)
    t0 = time.time()
    
    fao_parquet = "/Users/sezgintunca/_Coding_local/_Databases/FAO/raw_datasets/research_ready/country_annual_summary.parquet"
    if os.path.exists(fao_parquet):
        print(f"[*] Reading FAO country annual summary: {fao_parquet}...")
        table = pq.read_table(fao_parquet)
        df_fao = table.to_pandas()
        
        df_fao['Country_ISO3'] = df_fao['Country_ISO3'].astype(str).str.strip().str.upper()
        df_eu_fao = df_fao[df_fao['Country_ISO3'].isin(focal_iso)].copy()
        
        # Calculate Apparent Consumption (Production + Imports - Exports) and Self-Sufficiency Rate
        df_eu_fao['Apparent_Consumption_Tonnes'] = (
            df_eu_fao['Total_Production_Tonnes'] + df_eu_fao['Imports_Tonnes'] - df_eu_fao['Exports_Tonnes']
        )
        df_eu_fao['Apparent_Consumption_Tonnes'] = df_eu_fao['Apparent_Consumption_Tonnes'].clip(lower=0)
        
        df_eu_fao['Self_Sufficiency_Pct'] = np.where(
            df_eu_fao['Apparent_Consumption_Tonnes'] > 0,
            (df_eu_fao['Total_Production_Tonnes'] / df_eu_fao['Apparent_Consumption_Tonnes']) * 100.0,
            np.nan
        )
        
        df_eu_fao['Net_Trade_Balance_Tonnes'] = df_eu_fao['Exports_Tonnes'] - df_eu_fao['Imports_Tonnes']
        df_eu_fao['Country_Name_Mapped'] = df_eu_fao['Country_ISO3'].map(name_lookup)
        
        out_fao = os.path.join(DATA_DIR, "eu_fao_consumption_self_sufficiency_1950_2024.csv")
        df_eu_fao.to_csv(out_fao, index=False)
        print(f"[*] Saved FAO European consumption & self-sufficiency panel: {out_fao}")
        print(f"    Records: {len(df_eu_fao):,}, Countries: {df_eu_fao['Country_ISO3'].nunique()}, Years: {df_eu_fao['Year'].min()}-{df_eu_fao['Year'].max()}")
    else:
        print(f"[!] Warning: FAO dataset not found at {fao_parquet}.")

    # FAO Border Rejections
    fao_br_parquet = "/Users/sezgintunca/_Coding_local/_Databases/FAO/raw_datasets/research_ready/border_rejections.parquet"
    if os.path.exists(fao_br_parquet):
        print(f"[*] Reading FAO border rejections: {fao_br_parquet}...")
        df_br = pd.read_parquet(fao_br_parquet)
        df_br['Importing_Country_ISO3'] = df_br['Importing_Country_ISO3'].astype(str).str.strip().str.upper()
        df_br_eu = df_br[df_br['Importing_Country_ISO3'].isin(focal_iso)].copy()
        out_br = os.path.join(DATA_DIR, "eu_fao_border_rejections.csv")
        df_br_eu.to_csv(out_br, index=False)
        print(f"[*] Saved European border rejections: {out_br} ({len(df_br_eu):,} rows)")
        
    # FAO Primary Employment
    fao_emp_parquet = "/Users/sezgintunca/_Coding_local/_Databases/FAO/raw_datasets/research_ready/employment_primary_sector.parquet"
    if os.path.exists(fao_emp_parquet):
        print(f"[*] Reading FAO primary sector employment: {fao_emp_parquet}...")
        df_emp = pd.read_parquet(fao_emp_parquet)
        df_emp['Country_ISO3'] = df_emp['Country_ISO3'].astype(str).str.strip().str.upper()
        df_emp_eu = df_emp[df_emp['Country_ISO3'].isin(focal_iso)].copy()
        out_emp = os.path.join(DATA_DIR, "eu_fao_primary_employment.csv")
        df_emp_eu.to_csv(out_emp, index=False)
        print(f"[*] Saved European primary employment: {out_emp} ({len(df_emp_eu):,} rows)")
        
    print(f"[✓] Phase 3 completed in {time.time() - t0:.2f} seconds.")

def process_wto_tariff_data(focal_iso):
    print("\n" + "=" * 80)
    print("PHASE 4: EXTRACTING WTO APPLIED TARIFFS, PREFERENTIAL DUTIES & NTM NOTIFICATIONS")
    print("=" * 80)
    t0 = time.time()
    
    wto_mfn_csv = "/Users/sezgintunca/_Coding_local/_Databases/WTO_Workspace/data/wto_seafood_mfn_tariffs_multicountry_2015_2024.csv"
    wto_pref_csv = "/Users/sezgintunca/_Coding_local/_Databases/WTO_Workspace/data/wto_seafood_bilateral_preferential_tariffs.csv"
    wto_ntm_csv = "/Users/sezgintunca/_Coding_local/_Databases/WTO_Workspace/data/wto_seafood_eping_sps_tbt_notifications_all.csv"
    
    if os.path.exists(wto_mfn_csv):
        print(f"[*] Processing WTO MFN tariffs: {wto_mfn_csv}...")
        df_mfn = pd.read_csv(wto_mfn_csv)
        df_mfn['Reporter_Code'] = pd.to_numeric(df_mfn['Reporter_Code'], errors='coerce')
        tariff_summary = df_mfn.groupby(['Reporter_Name', 'Year', 'HS_Code'])['MFN_Applied_Duty_Pct'].mean().reset_index()
        out_mfn = os.path.join(DATA_DIR, "eu_wto_mfn_tariffs_by_species.csv")
        tariff_summary.to_csv(out_mfn, index=False)
        print(f"    Saved WTO MFN tariff summary: {out_mfn} ({len(tariff_summary):,} rows)")
    else:
        print("[!] WTO MFN dataset not found.")

    if os.path.exists(wto_pref_csv):
        print(f"[*] Processing WTO Preferential Tariffs: {wto_pref_csv}...")
        df_pref = pd.read_csv(wto_pref_csv)
        out_pref = os.path.join(DATA_DIR, "eu_wto_preferential_tariffs.csv")
        df_pref.to_csv(out_pref, index=False)
        print(f"    Saved WTO Preferential Tariffs: {out_pref} ({len(df_pref):,} rows)")

    if os.path.exists(wto_ntm_csv):
        print(f"[*] Processing WTO SPS/TBT notifications: {wto_ntm_csv}...")
        df_ntm = pd.read_csv(wto_ntm_csv)
        ntm_counts = df_ntm.groupby(['Notifying_Member', 'Area'])['Notification_ID'].count().reset_index()
        ntm_counts.rename(columns={'Notification_ID': 'Total_NTM_Notifications'}, inplace=True)
        out_ntm = os.path.join(DATA_DIR, "eu_wto_ntm_notifications_summary.csv")
        ntm_counts.to_csv(out_ntm, index=False)
        print(f"    Saved WTO NTM summary: {out_ntm} ({len(ntm_counts):,} rows)")
    else:
        print("[!] WTO NTM dataset not found.")
        
    print(f"[✓] Phase 4 completed in {time.time() - t0:.2f} seconds.")

def main():
    print("=" * 80)
    print("EUROPEAN SEAFOOD TRADE ARCHITECTURE: DATA EXTRACTION & HARMONIZATION")
    print("=" * 80)
    eu27_iso, focal_iso, cluster_lookup, status_lookup, name_lookup = load_mappings()
    print(f"Loaded {len(eu27_iso)} EU-27 member states and {len(focal_iso)} Focal European entities (including TUR, NOR, GBR, ISL, CHE).")
    
    process_species_data(eu27_iso, focal_iso, cluster_lookup, status_lookup, name_lookup)
    process_commodity_data(eu27_iso, focal_iso, cluster_lookup, status_lookup, name_lookup)
    process_fao_consumption_data(focal_iso, name_lookup)
    process_wto_tariff_data(focal_iso)
    
    print("\n" + "=" * 80)
    print("ALL 4 EXTRACTION PHASES COMPLETED SUCCESSFULLY! ALL DATASETS READY IN data/derived/")
    print("=" * 80)

if __name__ == "__main__":
    main()
