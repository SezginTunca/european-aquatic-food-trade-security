#!/usr/bin/env python3
"""
scripts/10_extra_european_gravity.py
====================================
Estimates Structural Poisson Pseudo-Maximum Likelihood (PPML) Gravity Models
for Extra-European Seafood Trade Corridors (Santos Silva & Tenreyro, 2006).

Provides the Supplementary Material econometric benchmark evaluating:
1. Geodesic intercontinental distance elasticity on cold-chain shipping.
2. Species trophic level valuation in trans-oceanic trade.
3. Processing form upgrading multipliers across supplier archetypes.
4. Multilateral resistance with importer fixed effects.

Output:
- tables/Table_SI_Extra_European_PPML_Gravity.csv
"""

import os
import sys
import math
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.genmod.generalized_linear_model import GLM
import pyarrow.dataset as ds

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "derived")
TABLES_DIR = os.path.join(PROJECT_ROOT, "tables")
CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")
os.makedirs(TABLES_DIR, exist_ok=True)

BILATERAL_PARQUET = os.path.join(DATA_DIR, "eu_species_bilateral_panel.parquet")

# Capital coordinates for European reporters and Top 20 extra-EU partners
CAPITALS = {
    # EU Reporters
    'AUT': (48.2082, 16.3738), 'BEL': (50.8503, 4.3517), 'BGR': (42.6977, 23.3219),
    'CYP': (35.1856, 33.3823), 'CZE': (50.0755, 14.4378), 'DEU': (52.5200, 13.4050),
    'DNK': (55.6761, 12.5683), 'ESP': (40.4168, -3.7038), 'EST': (59.4370, 24.7535),
    'FIN': (60.1699, 24.9384), 'FRA': (48.8566, 2.3522), 'GBR': (51.5074, -0.1278),
    'GRC': (37.9838, 23.7275), 'HRV': (45.8150, 15.9819), 'HUN': (47.4979, 19.0402),
    'IRL': (53.3498, -6.2603), 'ISL': (64.1466, -21.9426), 'ITA': (41.9028, 12.4964),
    'LTU': (54.6872, 25.2797), 'LUX': (49.8153, 6.1296), 'LVA': (56.9496, 24.1052),
    'MLT': (35.8989, 14.5146), 'NLD': (52.3676, 4.9041), 'NOR': (59.9139, 10.7522),
    'POL': (52.2297, 21.0122), 'PRT': (38.7223, -9.1393), 'ROU': (44.4268, 26.1025),
    'SVK': (48.1486, 17.1077), 'SVN': (46.0569, 14.5058), 'SWE': (59.3293, 18.0686),
    'TUR': (39.9334, 32.8597), 'CHE': (46.9480, 7.4474),
    # Top 20 Partners
    'CHN': (39.9042, 116.4074), 'ECU': (-0.1807, -78.4678), 'MAR': (34.0209, -6.8416),
    'VNM': (21.0285, 105.8542), 'USA': (38.9072, -77.0369), 'IND': (28.6139, 77.2090),
    'RUS': (55.7558, 37.6173), 'ARG': (-34.6037, -58.3816), 'GRL': (64.1814, -51.6941),
    'FRO': (62.0107, -6.7716), 'CAN': (45.4215, -75.6972), 'CHL': (-33.4489, -70.6693),
    'THA': (13.7563, 100.5018), 'PER': (-12.0464, -77.0428), 'IDN': (-6.2088, 106.8456),
    'NAM': (-22.5609, 17.0658), 'MRT': (18.0735, -15.9582), 'SYC': (-4.6191, 55.4513),
    'BGD': (23.8103, 90.4125), 'MUS': (-20.1609, 57.5012)
}

def haversine_km(c1, c2):
    if c1 is None or c2 is None:
        return 5000.0
    lat1, lon1 = math.radians(c1[0]), math.radians(c1[1])
    lat2, lon2 = math.radians(c2[0]), math.radians(c2[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2.0)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0)**2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return max(6371.0 * c, 100.0)

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

PROCESSING_LADDER = {
    'Live': 1,
    'Live / Fresh / Chilled Shellfish': 1,
    'Fresh / Chilled Whole Round': 2,
    'Frozen Whole Round': 3,
    'Fresh / Chilled Fillet & Meat': 4,
    'Frozen Fillet & Meat': 5,
    'Prepared / Preserved / Canned': 6,
    'Smoked / Cured / Dried / Salted': 6,
    'Flours, Meals & Pellets Fit for Human Consumption': 3,
    'Other Form': 3
}

def run_extra_european_gravity():
    print("=" * 80)
    print("STEP 10: STRUCTURAL PPML GRAVITY FOR EXTRA-EUROPEAN SEAFOOD CORRIDORS")
    print("=" * 80)

    # 1. Load Extra-European Bilateral Import Data (2000-2024)
    print(f"[*] Querying bilateral trade panel: {BILATERAL_PARQUET}")
    trade_dataset = ds.dataset(BILATERAL_PARQUET, format="parquet")
    trade_table = trade_dataset.to_table(
        filter=(ds.field('Year') >= 2000) & (ds.field('Year') <= 2024) &
               (ds.field('Trade_Flow') == 'Import') & (ds.field('Is_Extra_European') == True),
        columns=[
            'Year', 'Reporter_ISO3', 'Partner_ISO3', 'Species_Group',
            'Processing_Form', 'Trophic_Level', 'Trade_Value_USD', 'Weight_KG'
        ]
    )
    df = trade_table.to_pandas()
    df['Partner_ISO3'] = df['Partner_ISO3'].str.strip().str.upper()
    df['Reporter_ISO3'] = df['Reporter_ISO3'].str.strip().str.upper()

    # Filter to Top-20 partners
    top20_set = set(ARCHETYPE_LOOKUP.keys())
    df = df[df['Partner_ISO3'].isin(top20_set)].copy()

    # Aggregate at dyad-species-processing form level per year
    df_agg = df.groupby(['Year', 'Reporter_ISO3', 'Partner_ISO3', 'Species_Group', 'Processing_Form']).agg(
        Trade_Value_USD=('Trade_Value_USD', 'sum'),
        Weight_KG=('Weight_KG', 'sum'),
        Trophic_Level=('Trophic_Level', 'mean')
    ).reset_index()

    # Compute distance
    distances = []
    for _, row in df_agg.iterrows():
        c_imp = CAPITALS.get(row['Reporter_ISO3'])
        c_exp = CAPITALS.get(row['Partner_ISO3'])
        dist = haversine_km(c_imp, c_exp)
        distances.append(dist)
    df_agg['Distance_KM'] = distances
    df_agg['Log_Distance'] = np.log(df_agg['Distance_KM'])

    # Processing ladder
    df_agg['Processing_Form_Ordinal'] = df_agg['Processing_Form'].map(PROCESSING_LADDER).fillna(3)

    # Archetype dummies
    df_agg['Supplier_Archetype'] = df_agg['Partner_ISO3'].map(ARCHETYPE_LOOKUP)

    # Shock dummies
    df_agg['Post_2014_Russian_Ban'] = (df_agg['Year'] >= 2014).astype(int)
    df_agg['Post_2020_Brexit_COVID'] = (df_agg['Year'] >= 2020).astype(int)

    # Scale Trade Value in Thousands USD for PPML stability
    y_val = df_agg['Trade_Value_USD'] / 1e3

    # Model 1: Pooled PPML with Archetype Controls
    pooled_vars = [
        'Log_Distance', 'Trophic_Level', 'Processing_Form_Ordinal',
        'Post_2014_Russian_Ban', 'Post_2020_Brexit_COVID'
    ]
    arch_dummies = pd.get_dummies(df_agg['Supplier_Archetype'], prefix='Archetype', drop_first=True, dtype=float)
    X_pooled = sm.add_constant(pd.concat([df_agg[pooled_vars].astype(float), arch_dummies], axis=1))

    print("[*] Estimating Model 1: Pooled Extra-European PPML...")
    poisson_pooled = GLM(y_val, X_pooled, family=sm.families.Poisson()).fit(cov_type='HC1')

    # Model 2: Importer Fixed-Effects PPML (Structural Multilateral Resistance)
    imp_dummies = pd.get_dummies(df_agg['Reporter_ISO3'], prefix='Importer', drop_first=True, dtype=float)
    fe_vars = ['Log_Distance', 'Trophic_Level', 'Processing_Form_Ordinal', 'Post_2014_Russian_Ban', 'Post_2020_Brexit_COVID']
    X_fe = sm.add_constant(pd.concat([df_agg[fe_vars].astype(float), arch_dummies, imp_dummies], axis=1))

    print("[*] Estimating Model 2: Fixed-Effects Extra-European PPML...")
    poisson_fe = GLM(y_val, X_fe, family=sm.families.Poisson()).fit(cov_type='HC1')

    # Extract results
    def extract_ppml_table(model, model_name, features):
        rows = []
        conf = model.conf_int()
        for f in features:
            if f in model.params:
                coef = model.params[f]
                se = model.bse[f]
                zval = model.tvalues[f] if hasattr(model, 'tvalues') else coef/se
                pval = model.pvalues[f]
                ci_l = conf.loc[f, 0]
                ci_u = conf.loc[f, 1]
                rows.append({
                    'Model_Specification': model_name,
                    'Variable': f,
                    'PPML_Coefficient': round(coef, 4),
                    'Robust_Std_Error': round(se, 4),
                    'Z_Statistic': round(zval, 2),
                    'P_Value': round(pval, 4),
                    'CI_95_Lower': round(ci_l, 4),
                    'CI_95_Upper': round(ci_u, 4)
                })
        return rows

    report_features = pooled_vars + list(arch_dummies.columns)
    r1 = extract_ppml_table(poisson_pooled, "Pooled Extra-EU PPML", report_features)
    r2 = extract_ppml_table(poisson_fe, "Importer Fixed-Effects Extra-EU PPML", report_features)

    df_results = pd.DataFrame(r1 + r2)

    interp_map = {
        'Log_Distance': 'Distance Elasticity (ln km)',
        'Trophic_Level': 'Species Trophic Level Gradient',
        'Processing_Form_Ordinal': 'Value-Added Processing Form Ladder',
        'Post_2014_Russian_Ban': 'Post-2014 Sanctions Re-routing Dummy',
        'Post_2020_Brexit_COVID': 'Post-2020 Brexit & Pandemic Disruption',
        'Archetype_Cold-Water Farmed Salmonids': 'Archetype: Cold-Water Farmed Salmonids (vs Americas)',
        'Archetype_Global Processing Hubs': 'Archetype: Global Processing Hubs (vs Americas)',
        'Archetype_North Atlantic & Arctic Capture': 'Archetype: North Atlantic & Arctic Capture (vs Americas)',
        'Archetype_Tropical Farmed Aquaculture': 'Archetype: Tropical Farmed Aquaculture (vs Americas)',
        'Archetype_Upwelling & Ocean Tuna Fisheries': 'Archetype: Upwelling & Ocean Tuna Fisheries (vs Americas)'
    }
    df_results['Economic_Interpretation'] = df_results['Variable'].map(interp_map).fillna(df_results['Variable'])

    out_path = os.path.join(TABLES_DIR, "Table_SI_Extra_European_PPML_Gravity.csv")
    df_results.to_csv(out_path, index=False)
    print(f"[+] Saved Extra-European PPML Estimates to: {out_path}")
    print("\nSummary of Key Extra-European PPML Estimates (Fixed Effects):")
    fe_sub = df_results[df_results['Model_Specification'] == 'Importer Fixed-Effects Extra-EU PPML']
    print(fe_sub[['Variable', 'PPML_Coefficient', 'Robust_Std_Error', 'P_Value', 'Economic_Interpretation']].to_string(index=False))
    print("=" * 80)
    print("STEP 10 COMPLETED SUCCESSFULLY.")
    print("=" * 80)

if __name__ == "__main__":
    run_extra_european_gravity()
