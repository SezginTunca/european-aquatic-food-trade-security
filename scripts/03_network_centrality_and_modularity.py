#!/usr/bin/env python3
"""
scripts/03_network_centrality_and_modularity.py
===============================================
Advanced network topology and econometric gravity analytics for European seafood trade:
1. Bilateral network centrality: In/Out Degree, PageRank, Hubs/Authorities
2. Tripartite Value Chain Corridors:
   - Salmon Corridor: Norway -> Poland/Denmark/Lithuania -> Germany/France/UK
   - Mediterranean Seabass/Seabream Corridor: Türkiye & Greece -> Italy/Spain/Germany
3. Econometric Gravity / Shock Panel Estimation:
   - Evaluates Single Market effect, EEA effect, Customs Union effect
   - Structural shocks: 2004 Enlargement, 2014 Russia Ban, 2020 Brexit, 2020 COVID-19

Outputs in tables/:
- Table_7_Key_Bilateral_Value_Chain_Corridors.csv
- Table_8_Econometric_Gravity_Determinants.csv
"""

import os
import sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "derived")
TABLES_DIR = os.path.join(PROJECT_ROOT, "tables")
CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")
os.makedirs(TABLES_DIR, exist_ok=True)

SP_INTRA_PARQUET = os.path.join(DATA_DIR, "eu_species_intra_panel.parquet")
SP_FULL_PARQUET = os.path.join(DATA_DIR, "eu_species_bilateral_panel.parquet")
MAP_CSV = os.path.join(CONFIG_DIR, "european_country_mapping.csv")

def load_data():
    print("[*] Loading intra-European panel for network & value chain modeling...")
    df_intra = pq.read_table(SP_INTRA_PARQUET).to_pandas()
    print(f"    Loaded {len(df_intra):,} intra-European records.")
    
    df_map = pd.read_csv(MAP_CSV)
    df_map['ISO3'] = df_map['ISO3'].astype(str).str.strip().str.upper()
    return df_intra, df_map

def compute_network_centralities(df_intra, df_map):
    print("\n[+] Computing Network Topology & Centrality (PageRank & Hub/Authority)...")
    # Focus on modern period 2015-2025 exports
    sub = df_intra[(df_intra['Year'] >= 2015) & (df_intra['Trade_Flow'] == 'Export')].copy()
    
    # Bilateral adjacency matrix
    adj = sub.groupby(['Reporter_ISO3', 'Partner_ISO3'])['Trade_Value_USD'].sum().unstack(fill_value=0)
    all_nodes = sorted(list(set(adj.index).union(set(adj.columns))))
    adj = adj.reindex(index=all_nodes, columns=all_nodes, fill_value=0)
    
    matrix = adj.values
    n = len(all_nodes)
    
    # Compute Weighted Out-Degree (Exports) and In-Degree (Imports)
    out_degree = adj.sum(axis=1) / 1e6
    in_degree = adj.sum(axis=0) / 1e6
    
    # Unweighted Degree (Number of active partners)
    out_partners = (adj > 10000).sum(axis=1)
    in_partners = (adj > 10000).sum(axis=0)
    
    # Power iteration PageRank
    d = 0.85
    row_sums = matrix.sum(axis=1)
    M = np.zeros_like(matrix, dtype=float)
    for i in range(n):
        if row_sums[i] > 0:
            M[i, :] = matrix[i, :] / row_sums[i]
        else:
            M[i, :] = 1.0 / n
            
    v = np.ones(n) / n
    for _ in range(100):
        v = (1 - d) / n + d * np.dot(v, M)
        
    pagerank = pd.Series(v, index=all_nodes)
    
    # Kleinberg HITS (Hubs and Authorities)
    h = np.ones(n)
    a = np.ones(n)
    for _ in range(50):
        a = np.dot(matrix.T, h)
        a_norm = np.linalg.norm(a)
        if a_norm > 0:
            a = a / a_norm
            
        h = np.dot(matrix, a)
        h_norm = np.linalg.norm(h)
        if h_norm > 0:
            h = h / h_norm
            
    name_map = dict(zip(df_map['ISO3'], df_map['Country_Name']))
    cluster_map = dict(zip(df_map['ISO3'], df_map['European_Cluster']))
    status_map = dict(zip(df_map['ISO3'], df_map['Institutional_Status']))
    
    net_df = pd.DataFrame({
        'ISO3': all_nodes,
        'Country_Name': [name_map.get(iso, iso) for iso in all_nodes],
        'Institutional_Status': [status_map.get(iso, 'Other') for iso in all_nodes],
        'European_Cluster': [cluster_map.get(iso, 'Other') for iso in all_nodes],
        'Intra_EU_Export_USD_M': out_degree.round(1),
        'Intra_EU_Import_USD_M': in_degree.round(1),
        'Active_Export_Partners': out_partners.values,
        'Active_Import_Partners': in_partners.values,
        'PageRank_Centrality_Score': (pagerank * 100).round(3),
        'Authority_Sink_Score': pd.Series(a, index=all_nodes).round(4),
        'Hub_Source_Score': pd.Series(h, index=all_nodes).round(4)
    }).sort_values('Intra_EU_Export_USD_M', ascending=False)
    
    return net_df

def compute_table_7_value_chain_corridors(df_intra):
    print("\n[+] Computing Table 7: Key Bilateral Value Chain Corridors...")
    
    # 1. Salmon Tripartite Corridor
    salmon = df_intra[df_intra['Species_Group'] == 'Atlantic & Pacific Salmon'].copy()
    
    # Norwegian exports to EU
    nor_salmon = salmon[(salmon['Reporter_ISO3'] == 'NOR') & (salmon['Trade_Flow'] == 'Export')]
    nor_hubs = nor_salmon.groupby('Partner_ISO3')['Trade_Value_USD'].sum().sort_values(ascending=False) / 1e6
    
    # Polish re-exports to EU (Poland is Europe's primary smoked/filleted salmon hub)
    pol_salmon = salmon[(salmon['Reporter_ISO3'] == 'POL') & (salmon['Trade_Flow'] == 'Export')]
    pol_dests = pol_salmon.groupby('Partner_ISO3')['Trade_Value_USD'].sum().sort_values(ascending=False) / 1e6
    
    # Danish re-exports to EU
    dnk_salmon = salmon[(salmon['Reporter_ISO3'] == 'DNK') & (salmon['Trade_Flow'] == 'Export')]
    dnk_dests = dnk_salmon.groupby('Partner_ISO3')['Trade_Value_USD'].sum().sort_values(ascending=False) / 1e6
    
    # 2. Mediterranean Mariculture Corridor (European Seabass & Gilthead Seabream)
    bass_bream = df_intra[df_intra['Species_Group'].isin(['European Seabass', 'Gilthead Seabream'])].copy()
    
    # Turkish exports to EU
    tur_exports = bass_bream[(bass_bream['Reporter_ISO3'] == 'TUR') & (bass_bream['Trade_Flow'] == 'Export')]
    tur_dests = tur_exports.groupby(['Partner_ISO3', 'Species_Group'])['Trade_Value_USD'].sum().unstack(fill_value=0) / 1e6
    
    # Greek exports to EU
    grc_exports = bass_bream[(bass_bream['Reporter_ISO3'] == 'GRC') & (bass_bream['Trade_Flow'] == 'Export')]
    grc_dests = grc_exports.groupby(['Partner_ISO3', 'Species_Group'])['Trade_Value_USD'].sum().unstack(fill_value=0) / 1e6
    
    # Compile Corridor Records
    corridor_records = []
    
    # Top 5 Norway -> Hubs
    for p, val in nor_hubs.head(6).items():
        corridor_records.append({
            'Corridor_Type': 'Salmon Supply Pipeline',
            'Source_Country': 'Norway (NOR)',
            'Destination_Country': p,
            'Species_Group': 'Atlantic & Pacific Salmon',
            'Total_Trade_USD_M': round(val, 2),
            'Functional_Role': 'Upstream Farmed Producer -> Processing / Logistics Hub'
        })
        
    # Top 5 Poland -> End Markets
    for p, val in pol_dests.head(5).items():
        corridor_records.append({
            'Corridor_Type': 'Salmon Re-export / Processing',
            'Source_Country': 'Poland (POL)',
            'Destination_Country': p,
            'Species_Group': 'Atlantic & Pacific Salmon (Smoked/Fillets)',
            'Total_Trade_USD_M': round(val, 2),
            'Functional_Role': 'Value-Added Processing Hub -> Core Consumer Market'
        })
        
    # Top 5 Türkiye -> EU Markets
    tur_tot = tur_dests.sum(axis=1).sort_values(ascending=False)
    for p, val in tur_tot.head(6).items():
        corridor_records.append({
            'Corridor_Type': 'Mediterranean Seabass/Seabream Corridor',
            'Source_Country': 'Türkiye (TUR)',
            'Destination_Country': p,
            'Species_Group': 'European Seabass & Gilthead Seabream',
            'Total_Trade_USD_M': round(val, 2),
            'Functional_Role': 'Non-EU Aquaculture Powerhouse -> Mediterranean & Central EU'
        })
        
    # Top 5 Greece -> EU Markets
    grc_tot = grc_dests.sum(axis=1).sort_values(ascending=False)
    for p, val in grc_tot.head(5).items():
        corridor_records.append({
            'Corridor_Type': 'Mediterranean Seabass/Seabream Corridor',
            'Source_Country': 'Greece (GRC)',
            'Destination_Country': p,
            'Species_Group': 'European Seabass & Gilthead Seabream',
            'Total_Trade_USD_M': round(val, 2),
            'Functional_Role': 'Intra-EU Aquaculture Producer -> Intra-Single Market Consumers'
        })
        
    # Top 4 Iceland -> EU/UK Markets (Cod & Whitefish)
    isl_cod = df_intra[(df_intra['Reporter_ISO3'] == 'ISL') & (df_intra['Species_Group'] == 'Cod, Hake, Pollock & Whitefish') & (df_intra['Trade_Flow'] == 'Export')]
    isl_dests = isl_cod.groupby('Partner_ISO3')['Trade_Value_USD'].sum().sort_values(ascending=False) / 1e6
    for p, val in isl_dests.head(4).items():
        corridor_records.append({
            'Corridor_Type': 'North Atlantic Whitefish Corridor',
            'Source_Country': 'Iceland (ISL)',
            'Destination_Country': p,
            'Species_Group': 'Cod, Hake, Pollock & Whitefish',
            'Total_Trade_USD_M': round(val, 2),
            'Functional_Role': 'North Atlantic Demersal Harvester -> European Processing/Retail'
        })

    # Top 4 UK -> EU Markets (Langoustines, Pelagics, Salmon)
    uk_exp = df_intra[(df_intra['Reporter_ISO3'] == 'GBR') & (df_intra['Trade_Flow'] == 'Export')]
    uk_dests = uk_exp.groupby('Partner_ISO3')['Trade_Value_USD'].sum().sort_values(ascending=False) / 1e6
    for p, val in uk_dests.head(4).items():
        corridor_records.append({
            'Corridor_Type': 'UK-EU Post-TCA Trade Corridor',
            'Source_Country': 'United Kingdom (GBR)',
            'Destination_Country': p,
            'Species_Group': 'Pelagics, Crustaceans & Scottish Salmon',
            'Total_Trade_USD_M': round(val, 2),
            'Functional_Role': 'Key External Neighbor -> Cross-Channel EU Gateways'
        })
        
    table7 = pd.DataFrame(corridor_records)
    out_path = os.path.join(TABLES_DIR, "Table_7_Key_Bilateral_Value_Chain_Corridors.csv")
    table7.to_csv(out_path, index=False)
    print(f"    Saved Table 7: {out_path} ({len(table7)} rows)")
    return table7

def compute_table_8_econometric_gravity(df_intra):
    print("\n[+] Estimating Econometric Gravity Determinants of European Seafood Trade...")
    # Prepare annual bilateral dyad panel
    sub = df_intra[df_intra['Trade_Flow'] == 'Export'].copy()
    
    dyad_annual = sub.groupby(['Year', 'Reporter_ISO3', 'Partner_ISO3']).agg(
        Trade_Value_USD=('Trade_Value_USD', 'sum'),
        Weight_KG=('Weight_KG', 'sum'),
        Is_Intra_EU27=('Is_Intra_EU27', 'first'),
        Is_Norway_Corridor=('Is_Norway_Corridor', 'first'),
        Is_Turkey_Corridor=('Is_Turkey_Corridor', 'first'),
        Is_Iceland_Corridor=('Is_Iceland_Corridor', 'first'),
        Is_UK_Corridor=('Is_UK_Corridor', 'first')
    ).reset_index()
    
    # Log trade value
    dyad_annual = dyad_annual[dyad_annual['Trade_Value_USD'] > 1000].copy()
    dyad_annual['ln_Trade_Value'] = np.log(dyad_annual['Trade_Value_USD'])
    
    # Structural shock dummy variables
    dyad_annual['Post_2004_Enlargement'] = (dyad_annual['Year'] >= 2004).astype(int)
    dyad_annual['Post_2014_Russian_Ban'] = (dyad_annual['Year'] >= 2014).astype(int)
    dyad_annual['Post_2020_Brexit_COVID'] = (dyad_annual['Year'] >= 2020).astype(int)
    
    # Institutional dummies
    dyad_annual['Intra_EU27_Dummy'] = dyad_annual['Is_Intra_EU27'].astype(int)
    dyad_annual['Norway_Corridor_Dummy'] = dyad_annual['Is_Norway_Corridor'].astype(int)
    dyad_annual['Turkey_Corridor_Dummy'] = dyad_annual['Is_Turkey_Corridor'].astype(int)
    dyad_annual['Iceland_Corridor_Dummy'] = dyad_annual['Is_Iceland_Corridor'].astype(int)
    dyad_annual['UK_Corridor_Dummy'] = dyad_annual['Is_UK_Corridor'].astype(int)
    
    # Regression model: OLS with robust standard errors
    formula = (
        "ln_Trade_Value ~ Intra_EU27_Dummy + Norway_Corridor_Dummy + Turkey_Corridor_Dummy + "
        "Iceland_Corridor_Dummy + UK_Corridor_Dummy + "
        "Post_2004_Enlargement + Post_2014_Russian_Ban + Post_2020_Brexit_COVID"
    )
    
    model = smf.ols(formula, data=dyad_annual).fit(cov_type='HC1')
    
    params = model.params
    pvalues = model.pvalues
    conf_int = model.conf_int()
    
    records = []
    labels = {
        'Intercept': 'Baseline Constant (ln USD)',
        'Intra_EU27_Dummy': 'Intra-EU27 Single Market Integration Effect',
        'Norway_Corridor_Dummy': 'Norway-EU Salmonid Corridor Trade Effect',
        'Turkey_Corridor_Dummy': 'Türkiye-EU Customs Union & Aquaculture Effect',
        'Iceland_Corridor_Dummy': 'Iceland-EU North Atlantic EFTA Effect',
        'UK_Corridor_Dummy': 'UK-EU Bilateral Cross-Channel Trade Effect',
        'Post_2004_Enlargement': '2004 Eastern Enlargement Shock (10 New Member States)',
        'Post_2014_Russian_Ban': '2014 Russian Countersanctions Trade Re-routing',
        'Post_2020_Brexit_COVID': '2020 Brexit Transition & COVID-19 Disruption'
    }
    
    for var in params.index:
        coef = params[var]
        pval = pvalues[var]
        stars = '***' if pval < 0.001 else ('**' if pval < 0.01 else ('*' if pval < 0.05 else ''))
        records.append({
            'Variable': var,
            'Economic_Interpretation': labels.get(var, var),
            'Coefficient': round(coef, 4),
            'Std_Error': round(model.bse[var], 4),
            't_Statistic': round(model.tvalues[var], 2),
            'p_Value': f"{pval:.4f}{stars}",
            'Conf_Int_95_Lower': round(conf_int.loc[var, 0], 4),
            'Conf_Int_95_Upper': round(conf_int.loc[var, 1], 4)
        })
        
    table8 = pd.DataFrame(records)
    table8.attrs['R_squared'] = round(model.rsquared, 4)
    table8.attrs['N_obs'] = int(model.nobs)
    
    out_path = os.path.join(TABLES_DIR, "Table_8_Econometric_Gravity_Determinants.csv")
    table8.to_csv(out_path, index=False)
    print(f"    Saved Table 8: {out_path} (R² = {model.rsquared:.4f}, N = {int(model.nobs):,})")
    return table8

def main():
    print("=" * 80)
    print("ADVANCED NETWORK TOPOLOGY & GRAVITY ECONOMETRICS FOR EUROPEAN SEAFOOD")
    print("=" * 80)
    
    df_intra, df_map = load_data()
    
    net_df = compute_network_centralities(df_intra, df_map)
    out_net = os.path.join(TABLES_DIR, "European_Bilateral_Network_Centrality_2015_2025.csv")
    net_df.to_csv(out_net, index=False)
    print(f"    Saved Network Centralities: {out_net} ({len(net_df)} nodes)")
    
    compute_table_7_value_chain_corridors(df_intra)
    compute_table_8_econometric_gravity(df_intra)
    
    print("\n" + "=" * 80)
    print("NETWORK & GRAVITY ANALYTICS COMPLETED SUCCESSFULLY! (Tables 7-8 in tables/)")
    print("=" * 80)

if __name__ == "__main__":
    main()
