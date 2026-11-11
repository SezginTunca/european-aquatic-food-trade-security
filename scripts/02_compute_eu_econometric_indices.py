#!/usr/bin/env python3
"""
scripts/02_compute_eu_econometric_indices.py
============================================
Computes publication-grade econometric, trophic, and structural trade indices:
1. Balassa Revealed Comparative Advantage (RCA) across 17 species groups & 4 commodity categories
2. Bilateral and Multilateral Grubel-Lloyd Intra-Industry Trade (IIT)
3. Horizontal vs. Vertical IIT Decomposition (GHM method with alpha = 0.15)
4. Mean Trophic Levels of Exports (MTL_X), Imports (MTL_M), and Ecological Balance (Delta_MTL)
5. Herfindahl-Hirschman Market Concentration Indices (HHI) for exports and imports
6. Intra-European vs. Extra-European Trade Matrix by Policy Era

Outputs in tables/:
- Table_1_European_Seafood_Macro_Panel_1962_2025.csv
- Table_2_Species_Level_Trade_and_Unit_Values_1988_2025.csv
- Table_3_Intra_vs_Extra_European_Trade_Matrix.csv
- Table_4_Balassa_RCA_and_Specialization.csv
- Table_5_Grubel_Lloyd_IIT_and_Quality_Decomposition.csv
- Table_6_Trophic_Levels_and_Ecological_Trade_Balances.csv
"""

import os
import sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "derived")
TABLES_DIR = os.path.join(PROJECT_ROOT, "tables")
CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")
os.makedirs(TABLES_DIR, exist_ok=True)

SP_PARQUET = os.path.join(DATA_DIR, "eu_species_bilateral_panel.parquet")
CM_PARQUET = os.path.join(DATA_DIR, "eu_commodity_sitc_panel_1962_2025.parquet")
MAP_CSV = os.path.join(CONFIG_DIR, "european_country_mapping.csv")

def load_data():
    print("[*] Loading European species bilateral panel...")
    df_sp = pq.read_table(SP_PARQUET).to_pandas()
    print(f"    Loaded {len(df_sp):,} species microdata records.")
    
    print("[*] Loading European commodity SITC panel...")
    df_cm = pq.read_table(CM_PARQUET).to_pandas()
    print(f"    Loaded {len(df_cm):,} commodity longitudinal records.")
    
    df_map = pd.read_csv(MAP_CSV)
    df_map['ISO3'] = df_map['ISO3'].astype(str).str.strip().str.upper()
    focal_iso = set(df_map[df_map['Is_Focal_European']]['ISO3'].unique())
    eu27_iso = set(df_map[df_map['Is_EU27']]['ISO3'].unique())
    
    return df_sp, df_cm, df_map, focal_iso, eu27_iso

def compute_table_1_macro_panel(df_cm):
    print("\n[+] Computing Table 1: European Seafood Macro Panel (1962-2025)...")
    # Aggregate across years and eras
    macro = df_cm.groupby(['Year', 'Era', 'Era_Label', 'Trade_Flow']).agg(
        Total_Value_USD=('Trade_Value_USD', 'sum'),
        Total_Weight_KG=('Weight_KG', 'sum'),
        Reporters_Count=('Reporter_ISO3', 'nunique'),
        Partners_Count=('Partner_ISO3', 'nunique')
    ).reset_index()
    
    pivot = macro.pivot_table(
        index=['Year', 'Era', 'Era_Label'],
        columns='Trade_Flow',
        values=['Total_Value_USD', 'Total_Weight_KG', 'Reporters_Count', 'Partners_Count']
    ).reset_index()
    
    pivot.columns = ['_'.join(col).strip('_') if col[1] else col[0] for col in pivot.columns]
    
    x_val = pivot.get('Total_Value_USD_Export', 0)
    m_val = pivot.get('Total_Value_USD_Import', 0)
    x_wgt = pivot.get('Total_Weight_KG_Export', 0)
    m_wgt = pivot.get('Total_Weight_KG_Import', 0)
    
    table1 = pd.DataFrame({
        'Year': pivot['Year'],
        'Era': pivot['Era'],
        'Era_Label': pivot['Era_Label'],
        'Export_Value_USD_B': (x_val / 1e9).round(4),
        'Import_Value_USD_B': (m_val / 1e9).round(4),
        'Total_Turnover_USD_B': ((x_val + m_val) / 1e9).round(4),
        'Net_Trade_Balance_USD_B': ((x_val - m_val) / 1e9).round(4),
        'Net_Trade_Ratio': np.where((x_val + m_val) > 0, ((x_val - m_val) / (x_val + m_val)).round(4), np.nan),
        'Export_Weight_MMT': (x_wgt / 1e9).round(4),
        'Import_Weight_MMT': (m_wgt / 1e9).round(4),
        'Export_Unit_Value_USD_KG': np.where(x_wgt > 0, (x_val / x_wgt).round(2), np.nan),
        'Import_Unit_Value_USD_KG': np.where(m_wgt > 0, (m_val / m_wgt).round(2), np.nan),
        'Grubel_Lloyd_IIT': np.where((x_val + m_val) > 0, (1 - abs(x_val - m_val) / (x_val + m_val)).round(4), np.nan),
        'Active_Reporters': pivot.get('Reporters_Count_Export', 0).astype(int),
        'Active_Partners': pivot.get('Partners_Count_Import', 0).astype(int)
    })
    
    out_path = os.path.join(TABLES_DIR, "Table_1_European_Seafood_Macro_Panel_1962_2025.csv")
    table1.to_csv(out_path, index=False)
    print(f"    Saved Table 1: {out_path} ({len(table1)} rows)")
    return table1

def compute_table_2_species_trade(df_sp):
    print("\n[+] Computing Table 2: Species-Level Trade Dynamics & Unit Values (1988-2025)...")
    sp_summary = df_sp.groupby(['Species_Group', 'Trade_Flow']).agg(
        Total_Value_USD=('Trade_Value_USD', 'sum'),
        Total_Weight_KG=('Weight_KG', 'sum')
    ).reset_index()
    
    # Pre-calculate overall species mean trophic level
    sp_mtl_map = df_sp.groupby('Species_Group')['Trophic_Level'].mean().to_dict()
    
    piv = sp_summary.pivot_table(
        index='Species_Group',
        columns='Trade_Flow',
        values=['Total_Value_USD', 'Total_Weight_KG']
    ).reset_index()
    
    piv.columns = ['_'.join(col).strip('_') if col[1] else col[0] for col in piv.columns]
    
    x_val = piv.get('Total_Value_USD_Export', 0)
    m_val = piv.get('Total_Value_USD_Import', 0)
    x_wgt = piv.get('Total_Weight_KG_Export', 0)
    m_wgt = piv.get('Total_Weight_KG_Import', 0)
    
    # Calculate standard deviation of unit prices for error bars
    if 'Unit_Price_USD_per_KG' not in df_sp.columns:
        df_sp['Unit_Price_USD_per_KG'] = np.where(df_sp['Weight_KG'] > 0, df_sp['Trade_Value_USD'] / df_sp['Weight_KG'], np.nan)
        
    sp_summary_sd = df_sp.groupby(['Species_Group', 'Trade_Flow'])['Unit_Price_USD_per_KG'].std().reset_index()
    piv_sd = sp_summary_sd.pivot_table(index='Species_Group', columns='Trade_Flow', values='Unit_Price_USD_per_KG').reset_index()
    
    x_sd = piv_sd.get('Export', pd.Series(0, index=piv_sd.index)).fillna(0)
    m_sd = piv_sd.get('Import', pd.Series(0, index=piv_sd.index)).fillna(0)
    
    tot_val = x_val + m_val
    
    table2 = pd.DataFrame({
        'Species_Group': piv['Species_Group'],
        'Mean_Trophic_Level': piv['Species_Group'].map(sp_mtl_map).round(2),
        'Total_Turnover_USD_B': (tot_val / 1e9).round(3),
        'Export_Value_USD_B': (x_val / 1e9).round(3),
        'Import_Value_USD_B': (m_val / 1e9).round(3),
        'Share_of_Total_Turnover_Pct': ((tot_val / tot_val.sum()) * 100).round(2),
        'Export_Weight_MMT': (x_wgt / 1e9).round(3),
        'Import_Weight_MMT': (m_wgt / 1e9).round(3),
        'Export_Unit_Price_USD_KG': np.where(x_wgt > 0, (x_val / x_wgt).round(2), np.nan),
        'Import_Unit_Price_USD_KG': np.where(m_wgt > 0, (m_val / m_wgt).round(2), np.nan),
        'Export_Unit_Price_SD': x_sd.round(2),
        'Import_Unit_Price_SD': m_sd.round(2),
        'Net_Trade_Ratio': np.where(tot_val > 0, ((x_val - m_val) / tot_val).round(3), np.nan),
        'Grubel_Lloyd_IIT': np.where(tot_val > 0, (1 - abs(x_val - m_val) / tot_val).round(3), np.nan)
    }).sort_values('Total_Turnover_USD_B', ascending=False)
    
    out_path = os.path.join(TABLES_DIR, "Table_2_Species_Level_Trade_and_Unit_Values_1988_2025.csv")
    table2.to_csv(out_path, index=False)
    print(f"    Saved Table 2: {out_path} ({len(table2)} rows)")
    return table2

def compute_table_3_intra_vs_extra(df_sp):
    print("\n[+] Computing Table 3: Intra-European vs. Extra-European Trade Matrix by Policy Era...")
    # Define eras for the 1988-2025 period
    def assign_era(yr):
        if yr <= 1999:
            return "Era 2: 1988-1999 (Internal Market & Early Aquaculture)"
        elif yr <= 2014:
            return "Era 3: 2000-2014 (Eastern Enlargement & SPS Modernization)"
        else:
            return "Era 4: 2015-2025 (Multipolarity, Shocks & Re-alignment)"
            
    df_sp['Modern_Era'] = df_sp['Year'].apply(assign_era)
    
    # Classify flows
    def flow_channel(row):
        is_intra_eu27 = row['Is_Intra_EU27']
        is_nor = row.get('Is_Norway_Corridor', False)
        is_tur = row.get('Is_Turkey_Corridor', False)
        is_isl = row.get('Is_Iceland_Corridor', False)
        is_uk = row.get('Is_UK_Corridor', False)
        is_che = row.get('Is_Switzerland_Corridor', False)
        is_intra = row['Is_Intra_European']
        
        if is_intra_eu27:
            return "Intra-EU27 Single Market"
        elif is_nor:
            return "EU-Norway Salmonid Corridor"
        elif is_tur:
            return "EU-Türkiye Mediterranean Corridor"
        elif is_isl:
            return "EU-Iceland North Atlantic Corridor"
        elif is_uk:
            return "EU-UK Trade & Cooperation Corridor"
        elif is_che:
            return "EU-Switzerland Alpine Corridor"
        elif is_intra:
            return "Other Intra-European (EFTA / Regional)"
        else:
            return "Extra-European Rest of World"
            
    df_sp['Trade_Channel'] = df_sp.apply(flow_channel, axis=1)
    
    agg = df_sp.groupby(['Modern_Era', 'Trade_Channel', 'Trade_Flow']).agg(
        Value_USD=('Trade_Value_USD', 'sum'),
        Weight_KG=('Weight_KG', 'sum')
    ).reset_index()
    
    piv = agg.pivot_table(
        index=['Modern_Era', 'Trade_Channel'],
        columns='Trade_Flow',
        values=['Value_USD', 'Weight_KG']
    ).reset_index()
    
    piv.columns = ['_'.join(col).strip('_') if col[1] else col[0] for col in piv.columns]
    
    x_val = piv.get('Value_USD_Export', 0)
    m_val = piv.get('Value_USD_Import', 0)
    x_wgt = piv.get('Weight_KG_Export', 0)
    m_wgt = piv.get('Weight_KG_Import', 0)
    
    tot_val = x_val + m_val
    table3 = pd.DataFrame({
        'Policy_Era': piv['Modern_Era'],
        'Trade_Channel': piv['Trade_Channel'],
        'Turnover_USD_B': (tot_val / 1e9).round(3),
        'Export_USD_B': (x_val / 1e9).round(3),
        'Import_USD_B': (m_val / 1e9).round(3),
        'Export_Weight_MMT': (x_wgt / 1e9).round(3),
        'Import_Weight_MMT': (m_wgt / 1e9).round(3),
        'Export_Unit_Price_USD_KG': np.where(x_wgt > 0, (x_val / x_wgt).round(2), np.nan),
        'Import_Unit_Price_USD_KG': np.where(m_wgt > 0, (m_val / m_wgt).round(2), np.nan),
        'Grubel_Lloyd_IIT': np.where(tot_val > 0, (1 - abs(x_val - m_val) / tot_val).round(3), np.nan)
    }).sort_values(['Policy_Era', 'Turnover_USD_B'], ascending=[True, False])
    
    out_path = os.path.join(TABLES_DIR, "Table_3_Intra_vs_Extra_European_Trade_Matrix.csv")
    table3.to_csv(out_path, index=False)
    print(f"    Saved Table 3: {out_path} ({len(table3)} rows)")
    return table3

def compute_table_4_balassa_rca(df_sp, focal_iso):
    print("\n[+] Computing Table 4: Balassa Revealed Comparative Advantage (RCA) & Specialization...")
    # Consider modern period (2015-2025) for current comparative advantage
    df_recent = df_sp[(df_sp['Year'] >= 2015) & (df_sp['Trade_Flow'] == 'Export')].copy()
    
    # Calculate country total exports and species exports
    country_tot = df_recent.groupby('Reporter_ISO3')['Trade_Value_USD'].sum().to_dict()
    country_sp = df_recent.groupby(['Reporter_ISO3', 'Species_Group'])['Trade_Value_USD'].sum().reset_index()
    
    # Benchmark total across all European focal reporters
    benchmark_sp = df_recent.groupby('Species_Group')['Trade_Value_USD'].sum().to_dict()
    benchmark_tot = df_recent['Trade_Value_USD'].sum()
    
    rca_records = []
    top_species = [
        'Atlantic & Pacific Salmon',
        'Rainbow & Freshwater Trout',
        'European Seabass',
        'Gilthead Seabream',
        'Cod, Hake, Pollock & Whitefish',
        'Herring, Sardines & Small Pelagics',
        'Shrimp & Prawns',
        'Squid, Cuttlefish & Octopus',
        'Mussels, Oysters & Bivalves'
    ]
    
    for iso in sorted(focal_iso):
        x_i_tot = country_tot.get(iso, 0)
        if x_i_tot < 1e7: # Skip tiny micro-exporters (<$10M total exports over 2015-2025)
            continue
            
        c_sub = country_sp[country_sp['Reporter_ISO3'] == iso]
        c_sp_dict = dict(zip(c_sub['Species_Group'], c_sub['Trade_Value_USD']))
        
        row_dict = {'Reporter_ISO3': iso, 'Total_Export_USD_M': round(x_i_tot / 1e6, 1)}
        
        sp_col_map = {
            'Atlantic & Pacific Salmon': 'RCA_Atlantic_Salmon',
            'Rainbow & Freshwater Trout': 'RCA_Rainbow_Trout',
            'European Seabass': 'RCA_European_Seabass',
            'Gilthead Seabream': 'RCA_Gilthead_Seabream',
            'Cod, Hake, Pollock & Whitefish': 'RCA_Cod_and_Whitefish',
            'Herring, Sardines & Small Pelagics': 'RCA_Small_Pelagics_Herring',
            'Shrimp & Prawns': 'RCA_Shrimp_and_Prawns',
            'Squid, Cuttlefish & Octopus': 'RCA_Squid_and_Cephalopods',
            'Mussels, Oysters & Bivalves': 'RCA_Bivalves_and_Mussels'
        }
        for sp in top_species:
            x_i_s = c_sp_dict.get(sp, 0)
            x_ref_s = benchmark_sp.get(sp, 0)
            
            # Exclude focal country from the benchmark to avoid compressing dominant players to 1.0
            x_ref_s_excl = max(x_ref_s - x_i_s, 1.0)
            benchmark_tot_excl = max(benchmark_tot - x_i_tot, 1.0)
            
            if x_i_tot > 0 and benchmark_tot_excl > 0:
                rca = (x_i_s / x_i_tot) / (x_ref_s_excl / benchmark_tot_excl)
            else:
                rca = 0.0
            col_name = sp_col_map.get(sp, f"RCA_{sp.replace(' ', '_')}")
            row_dict[col_name] = round(rca, 2)
            
        rca_records.append(row_dict)
        
    df_rca = pd.DataFrame(rca_records).sort_values('Total_Export_USD_M', ascending=False)
    out_path = os.path.join(TABLES_DIR, "Table_4_Balassa_RCA_and_Specialization.csv")
    df_rca.to_csv(out_path, index=False)
    print(f"    Saved Table 4: {out_path} ({len(df_rca)} countries)")
    return df_rca

def compute_table_5_iit_decomposition(df_sp):
    print("\n[+] Computing Table 5: Grubel-Lloyd IIT & Quality Decomposition (Horizontal vs Vertical)...")
    # Bilateral pair level at species & year level
    # We aggregate flows by (Year, Reporter_ISO3, Partner_ISO3, Species_Group, Trade_Flow)
    sub = df_sp[df_sp['Is_Intra_European']].copy()
    
    pair_agg = sub.groupby(['Year', 'Reporter_ISO3', 'Partner_ISO3', 'Species_Group', 'Trade_Flow']).agg(
        Value_USD=('Trade_Value_USD', 'sum'),
        Weight_KG=('Weight_KG', 'sum')
    ).reset_index()
    
    piv = pair_agg.pivot_table(
        index=['Year', 'Reporter_ISO3', 'Partner_ISO3', 'Species_Group'],
        columns='Trade_Flow',
        values=['Value_USD', 'Weight_KG']
    ).reset_index()
    
    piv.columns = ['_'.join(col).strip('_') if col[1] else col[0] for col in piv.columns]
    
    # Filter for two-way trade (both Export and Import exist)
    x_val = piv.get('Value_USD_Export', 0).fillna(0)
    m_val = piv.get('Value_USD_Import', 0).fillna(0)
    x_wgt = piv.get('Weight_KG_Export', 0).fillna(0)
    m_wgt = piv.get('Weight_KG_Import', 0).fillna(0)
    
    two_way = piv[(x_val > 1000) & (m_val > 1000) & (x_wgt > 100) & (m_wgt > 100)].copy()
    two_way['X_USD'] = x_val[two_way.index]
    two_way['M_USD'] = m_val[two_way.index]
    two_way['X_KG'] = x_wgt[two_way.index]
    two_way['M_KG'] = m_wgt[two_way.index]
    
    two_way['UV_X'] = two_way['X_USD'] / two_way['X_KG']
    two_way['UV_M'] = two_way['M_USD'] / two_way['M_KG']
    two_way['Relative_UV'] = two_way['UV_X'] / two_way['UV_M']
    two_way['Total_Trade'] = two_way['X_USD'] + two_way['M_USD']
    two_way['Overlap_Trade'] = 2 * np.minimum(two_way['X_USD'], two_way['M_USD'])
    
    # Greenaway-Hine-Milner criteria with alpha = 0.15
    # HIIT: 0.85 <= Relative_UV <= 1.15
    # VIIT High: Relative_UV > 1.15
    # VIIT Low: Relative_UV < 0.85
    two_way['IIT_Type'] = np.where(
        (two_way['Relative_UV'] >= 0.85) & (two_way['Relative_UV'] <= 1.15),
        'Horizontal IIT (Same Quality Tier)',
        np.where(two_way['Relative_UV'] > 1.15, 'Vertical IIT (High Quality Export)', 'Vertical IIT (Low Quality Export)')
    )
    
    # Aggregate by 5-year periods
    two_way['Period'] = pd.cut(
        two_way['Year'],
        bins=[1987, 1995, 2005, 2015, 2025],
        labels=['1988-1995', '1996-2005', '2006-2015', '2016-2025']
    )
    
    decomp = two_way.groupby(['Period', 'IIT_Type'])['Overlap_Trade'].sum().unstack(fill_value=0)
    decomp['Total_IIT_Trade'] = decomp.sum(axis=1)
    
    for col in ['Horizontal IIT (Same Quality Tier)', 'Vertical IIT (High Quality Export)', 'Vertical IIT (Low Quality Export)']:
        if col in decomp.columns:
            decomp[f"{col}_Share_Pct"] = ((decomp[col] / decomp['Total_IIT_Trade']) * 100).round(2)
            
    decomp = decomp.reset_index()
    out_path = os.path.join(TABLES_DIR, "Table_5_Grubel_Lloyd_IIT_and_Quality_Decomposition.csv")
    decomp.to_csv(out_path, index=False)
    print(f"    Saved Table 5: {out_path}")
    return decomp

def compute_table_6_trophic_metabolism(df_sp, focal_iso, df_map):
    print("\n[+] Computing Table 6: Sovereign Trophic Levels & Ecological Trade Balances (2015-2025)...")
    df_recent = df_sp[df_sp['Year'] >= 2015].copy()
    
    status_map = dict(zip(df_map['ISO3'], df_map['Institutional_Status']))
    cluster_map = dict(zip(df_map['ISO3'], df_map['European_Cluster']))
    name_map = dict(zip(df_map['ISO3'], df_map['Country_Name']))
    
    records = []
    for iso in sorted(focal_iso):
        sub = df_recent[df_recent['Reporter_ISO3'] == iso]
        if sub.empty:
            continue
            
        exp = sub[sub['Trade_Flow'] == 'Export']
        imp = sub[sub['Trade_Flow'] == 'Import']
        
        x_val = exp['Trade_Value_USD'].sum()
        m_val = imp['Trade_Value_USD'].sum()
        
        if (x_val + m_val) < 1e7:
            continue
            
        mtl_x = np.average(exp['Trophic_Level'], weights=exp['Trade_Value_USD']) if (not exp.empty and x_val > 0) else np.nan
        mtl_m = np.average(imp['Trophic_Level'], weights=imp['Trade_Value_USD']) if (not imp.empty and m_val > 0) else np.nan
        
        # Intra vs Extra MTL
        exp_intra = exp[exp['Is_Intra_European']]
        imp_intra = imp[imp['Is_Intra_European']]
        mtl_x_intra = np.average(exp_intra['Trophic_Level'], weights=exp_intra['Trade_Value_USD']) if (not exp_intra.empty and exp_intra['Trade_Value_USD'].sum() > 0) else np.nan
        mtl_m_intra = np.average(imp_intra['Trophic_Level'], weights=imp_intra['Trade_Value_USD']) if (not imp_intra.empty and imp_intra['Trade_Value_USD'].sum() > 0) else np.nan
        
        records.append({
            'Reporter_ISO3': iso,
            'Country_Name': name_map.get(iso, iso),
            'Institutional_Status': status_map.get(iso, 'Other'),
            'European_Cluster': cluster_map.get(iso, 'Other'),
            'Export_Value_USD_M': round(x_val / 1e6, 1),
            'Import_Value_USD_M': round(m_val / 1e6, 1),
            'MTL_Export_Overall': round(mtl_x, 2) if not np.isnan(mtl_x) else np.nan,
            'MTL_Import_Overall': round(mtl_m, 2) if not np.isnan(mtl_m) else np.nan,
            'Ecological_Trophic_Spread': round(mtl_x - mtl_m, 2) if (not np.isnan(mtl_x) and not np.isnan(mtl_m)) else np.nan,
            'MTL_Export_Intra_EU': round(mtl_x_intra, 2) if not np.isnan(mtl_x_intra) else np.nan,
            'MTL_Import_Intra_EU': round(mtl_m_intra, 2) if not np.isnan(mtl_m_intra) else np.nan,
            'Trophic_Archetype': (
                'High-Trophic Exporter (Salmon/Pelagics)' if mtl_x >= 4.0 else (
                    'Mid-High Exporter (Seabass/Seabream)' if mtl_x >= 3.5 else 'Low/Mid Trophic / Mixed'
                )
            )
        })
        
    table6 = pd.DataFrame(records).sort_values('Export_Value_USD_M', ascending=False)
    out_path = os.path.join(TABLES_DIR, "Table_6_Trophic_Levels_and_Ecological_Trade_Balances.csv")
    table6.to_csv(out_path, index=False)
    print(f"    Saved Table 6: {out_path} ({len(table6)} countries)")
    return table6

def main():
    print("=" * 80)
    print("EUROPEAN SEAFOOD ECONOMETRIC ENGINE: COMPUTING INDICES & PUBLICATION TABLES")
    print("=" * 80)
    
    df_sp, df_cm, df_map, focal_iso, eu27_iso = load_data()
    
    compute_table_1_macro_panel(df_cm)
    compute_table_2_species_trade(df_sp)
    compute_table_3_intra_vs_extra(df_sp)
    compute_table_4_balassa_rca(df_sp, focal_iso)
    compute_table_5_iit_decomposition(df_sp)
    compute_table_6_trophic_metabolism(df_sp, focal_iso, df_map)
    
    print("\n" + "=" * 80)
    print("ALL ECONOMETRIC TABLES COMPUTED SUCCESSFULLY! (Tables 1-6 in tables/)")
    print("=" * 80)

if __name__ == "__main__":
    main()
