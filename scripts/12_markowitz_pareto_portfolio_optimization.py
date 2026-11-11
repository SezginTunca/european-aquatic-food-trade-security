#!/usr/bin/env python3
"""
scripts/12_markowitz_pareto_portfolio_optimization.py
======================================================
Publication-grade 600 DPI PNG and TrueType vector PDF generation script for
Constrained Multi-Objective Pareto and Markowitz Portfolio Optimization:

1. Evaluates multi-objective species-supplier portfolios across 16 species groups
   and 8 supplier archetypes under biosecurity, volatility, and budgetary constraints.
2. Generates Main Figure 9 (2x3 6-panel master optimization synthesis plate):
   - Figure_9 / Figure_9_Multi_Objective_Pareto_and_Markowitz_Optimization
3. Generates Supplementary Figure S11 (2x2 4-panel supplier archetype frontier & shock correlation plate):
   - Figure_S11 / Figure_S11_Supplier_Archetype_Markowitz_Frontier_and_Covariance
4. Generates Supplementary Figure S12 (2x2 4-panel sovereign two-stage stabilization & sensitivity plate):
   - Figure_S12 / Figure_S12_Sovereign_Two_Stage_Stabilization_and_Sensitivity

Outputs:
  figures/Figure_9.png (.pdf), figures/Figure_S11.png (.pdf), figures/Figure_S12.png (.pdf)
"""

import os
import sys
import shutil
import warnings
from decimal import Decimal, ROUND_HALF_UP
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

warnings.filterwarnings('ignore')
np.random.seed(42)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OPT_TABLES_DIR = os.path.join(PROJECT_ROOT, "Option1_Optimization", "tables")
TABLES_DIR = os.path.join(PROJECT_ROOT, "tables")
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "derived")
MAIN_FIG_DIR = os.path.join(PROJECT_ROOT, "figures")

for d in [TABLES_DIR, MAIN_FIG_DIR]:
    os.makedirs(d, exist_ok=True)

# Global High-Visibility Publication Typography (600 DPI, Vector PDF)
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11.5
plt.rcParams['axes.labelsize'] = 12.4
plt.rcParams['axes.titlesize'] = 13.2
plt.rcParams['xtick.labelsize'] = 10.8
plt.rcParams['ytick.labelsize'] = 10.8
plt.rcParams['legend.fontsize'] = 10.0
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['axes.linewidth'] = 1.05
plt.rcParams['grid.linewidth'] = 0.6
plt.rcParams['lines.linewidth'] = 1.8
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.02
plt.rcParams['text.parse_math'] = False

OKABE_ITO = {
    'blue': '#0072B2',
    'orange': '#E69F00',
    'green': '#009E73',
    'vermilion': '#D55E00',
    'purple': '#CC79A7',
    'sky': '#56B4E9',
    'yellow': '#F0E442',
    'navy': '#1E3A8A',
    'teal': '#0F766E',
    'crimson': '#B91C1C',
    'slate': '#334155',
    'grey': '#64748B',
    'black': '#0F172A'
}

DESCRIPTIVE_NAMES = {
    'Figure_9': 'Figure_9_Multi_Objective_Pareto_and_Markowitz_Optimization',
    'Figure_S11': 'Figure_S11_Supplier_Archetype_Markowitz_Frontier_and_Covariance',
    'Figure_S12': 'Figure_S12_Sovereign_Two_Stage_Stabilization_and_Sensitivity'
}

FULL_SPECIES_LABELS = {
    'Herring, Sardines & Small Pelagics': 'Herring, Sardines and Small Pelagics',
    'Mackerel & Horse Mackerel': 'Mackerel and Horse Mackerel',
    'Rainbow & Freshwater Trout': 'Rainbow and Freshwater Trout',
    'Squid, Cuttlefish & Octopus': 'Squid, Cuttlefish and Octopus',
    'Mussels, Oysters & Bivalves': 'Mussels, Oysters and Bivalves',
    'Flatfish & Turbot': 'Flatfish and Turbot',
    'Sea Cucumber, Urchins & Invertebrates': 'Sea Cucumber, Urchins and Invertebrates',
    'Shrimp & Prawns': 'Warm and Cold-Water Shrimp and Prawns',
    'Crabs, Lobsters & Other Crustaceans': 'Crabs, Lobsters and Other Crustaceans',
    'Gilthead Seabream': 'Gilthead Seabream',
    'European Seabass': 'European Seabass',
    'Pangasius, Tilapia & Freshwater Catfish': 'Pangasius, Tilapia and Freshwater Catfish',
    'Atlantic & Pacific Salmon': 'Atlantic and Pacific Salmon',
    'Bluefin Tuna & Major Tunas': 'Bluefin Tuna and Major Oceanic Tunas',
    'Other Marine & Freshwater Finfish': 'Other Marine and Freshwater Finfish',
    'Cod, Hake, Pollock & Whitefish': 'Cod, Hake, Pollock and Whitefish'
}

SUPPLIER_ARCHETYPES_MAP = {
    'Tropical Farmed Aquaculture': ['ECU', 'VNM', 'IND', 'IDN', 'BGD'],
    'North Atlantic & Arctic Capture': ['RUS', 'GRL'],
    'Cold-Water Farmed Salmonids': ['FRO', 'CHL'],
    'Upwelling & Ocean Tuna Fisheries': ['MAR', 'PER', 'NAM', 'MRT', 'SYC', 'MUS'],
    'Global Processing Hubs': ['CHN', 'THA'],
    'Americas Marine Capture': ['USA', 'ARG', 'CAN']
}
ARCHETYPE_LOOKUP = {iso: arch for arch, isos in SUPPLIER_ARCHETYPES_MAP.items() for iso in isos}

FULL_ARCHETYPE_LABELS = {
    'Intra-European Core': 'Intra-European Core Network',
    'Cold-Water Farmed Salmonids': 'Cold-Water Farmed Salmonids',
    'North Atlantic & Arctic Capture': 'North Atlantic and Arctic Capture',
    'Upwelling & Ocean Tuna Fisheries': 'Upwelling and Oceanic Tuna Fisheries',
    'Tropical Farmed Aquaculture': 'Tropical Farmed Aquaculture',
    'Other Third Countries': 'Other Global Partner Economies',
    'Global Processing Hubs': 'Global Reprocessing Hubs',
    'Americas Marine Capture': 'Americas Marine Wild Capture'
}


def fmt_pct_1(val: float, signed: bool = True) -> str:
    """Standard half-up 1-decimal percentage formatting so 30.45 -> +30.5% matching tables."""
    q = Decimal(str(val)).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)
    if signed and q > 0:
        return f"+{q}%"
    return f"{q}%"


def save_dual_figure(fig, fig_name):
    """Save 600 DPI PNG and vector PDF to figures/ under both numbered and descriptive names."""
    names_to_save = [fig_name]
    desc = DESCRIPTIVE_NAMES.get(fig_name)
    if desc and desc != fig_name:
        names_to_save.append(desc)

    primary_png = os.path.join(MAIN_FIG_DIR, f"{fig_name}.png")
    primary_pdf = os.path.join(MAIN_FIG_DIR, f"{fig_name}.pdf")

    fig.savefig(primary_png, dpi=600, bbox_inches='tight', pad_inches=0.02)
    fig.savefig(primary_pdf, format='pdf', bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)

    for name in names_to_save:
        dest_png = os.path.join(MAIN_FIG_DIR, f"{name}.png")
        dest_pdf = os.path.join(MAIN_FIG_DIR, f"{name}.pdf")
        if dest_png != primary_png:
            shutil.copy2(primary_png, dest_png)
        if dest_pdf != primary_pdf:
            shutil.copy2(primary_pdf, dest_pdf)

    print(f"  [✓] Saved {fig_name}.png/.pdf ({', '.join(names_to_save)}) to figures/.")


def sync_tables_and_compute_archetype_diagnostics():
    """Copy portfolio optimization CSVs to tables/ if separate, and compute exact 8x8 archetype shock correlation & optimal share table."""
    opt_csvs = [
        "Table_Opt1_Species_Model_Parameters.csv",
        "Table_Opt2_Regional_Portfolio_Optimization_Benchmarks.csv",
        "Table_Opt3_Species_Portfolio_Weights_Comparison.csv",
        "Table_Opt4_Pareto_Frontier_2D_Sweeps.csv",
        "Table_Opt5_Supplier_Archetype_Markowitz_Frontier.csv",
        "Table_Opt6_Sovereign_Efficiency_Gaps_32Countries.csv",
        "Table_Opt7_Shadow_Prices_and_Sensitivity.csv"
    ]
    if os.path.exists(OPT_TABLES_DIR) and OPT_TABLES_DIR != TABLES_DIR:
        print("[*] Synchronizing portfolio optimization CSV tables into tables/...")
        for fname in opt_csvs:
            src = os.path.join(OPT_TABLES_DIR, fname)
            dst = os.path.join(TABLES_DIR, fname)
            if os.path.exists(src) and src != dst:
                shutil.copy2(src, dst)
        print(f"  [✓] Copied portfolio optimization CSV tables to {TABLES_DIR}/")

    sp_params = pd.read_csv(os.path.join(TABLES_DIR, "Table_Opt1_Species_Model_Parameters.csv"))
    df_bench = pd.read_csv(os.path.join(TABLES_DIR, "Table_Opt2_Regional_Portfolio_Optimization_Benchmarks.csv"))
    df_weights = pd.read_csv(os.path.join(TABLES_DIR, "Table_Opt3_Species_Portfolio_Weights_Comparison.csv"))
    df_pareto = pd.read_csv(os.path.join(TABLES_DIR, "Table_Opt4_Pareto_Frontier_2D_Sweeps.csv"))
    df_arch = pd.read_csv(os.path.join(TABLES_DIR, "Table_Opt5_Supplier_Archetype_Markowitz_Frontier.csv"))
    df_sov = pd.read_csv(os.path.join(TABLES_DIR, "Table_Opt6_Sovereign_Efficiency_Gaps_32Countries.csv"))
    df_shad = pd.read_csv(os.path.join(TABLES_DIR, "Table_Opt7_Shadow_Prices_and_Sensitivity.csv"))

    archetypes = [
        'Intra-European Core',
        'Cold-Water Farmed Salmonids',
        'North Atlantic & Arctic Capture',
        'Upwelling & Ocean Tuna Fisheries',
        'Tropical Farmed Aquaculture',
        'Other Third Countries',
        'Global Processing Hubs',
        'Americas Marine Capture'
    ]
    edible_species = sorted(sp_params['Species_Group'].unique())
    t8_path = os.path.join(TABLES_DIR, "Table_Opt8_Supplier_Archetype_Correlation_and_Shares.csv")
    parquet_path = os.path.join(DATA_DIR, "eu_species_bilateral_panel.parquet")

    if os.path.exists(t8_path) and not os.path.exists(parquet_path):
        print(f"  [✓] Utilizing pre-computed archetype diagnostics from {os.path.basename(t8_path)}")
        arch_diag_df = pd.read_csv(t8_path)
        corr_cols = [f'Corr_{idx+1}' for idx in range(len(archetypes))]
        corr_arch = arch_diag_df[corr_cols].to_numpy()
    else:
        dataset = ds.dataset(parquet_path, format="parquet")
        df_tr = dataset.to_table(
            filter=(ds.field('Year') >= 1990) & (ds.field('Year') <= 2024) & (ds.field('Trade_Flow') == 'Import'),
            columns=['Year', 'Partner_ISO3', 'Species_Group', 'Processing_Form', 'Trade_Value_USD', 'Weight_KG', 'Is_Extra_European']
        ).to_pandas()
        df_tr = df_tr[df_tr['Species_Group'].isin(edible_species)].copy()
        df_tr['Partner_ISO3'] = df_tr['Partner_ISO3'].str.strip().str.upper()
        df_tr['Weight_Tonnes'] = df_tr['Weight_KG'] / 1e3
        lwe_map = {
            'Live': 1.00, 'Fresh / Chilled Whole Round': 1.05, 'Frozen Whole Round': 1.10,
            'Fresh / Chilled Fillet & Meat': 2.60, 'Frozen Fillet & Meat': 2.70,
            'Prepared / Preserved / Canned': 2.20, 'Smoked / Cured / Dried / Salted': 2.50,
            'Live / Fresh / Chilled Shellfish': 1.05, 'Flours, Meals & Pellets Fit for Human Consumption': 4.50,
            'Other Form': 1.40
        }
        df_tr['LWE_Tonnes'] = df_tr['Weight_Tonnes'] * df_tr['Processing_Form'].map(lwe_map).fillna(1.50)
        df_tr['Supplier_Archetype'] = np.where(
            ~df_tr['Is_Extra_European'],
            'Intra-European Core',
            df_tr['Partner_ISO3'].map(ARCHETYPE_LOOKUP).fillna('Other Third Countries')
        )

        years = list(range(1990, 2025))
        piv = df_tr.pivot_table(index='Year', columns='Supplier_Archetype', values='Weight_Tonnes', aggfunc='sum').reindex(index=years, columns=archetypes).fillna(0.0)
        t_vals = np.arange(len(years), dtype=float)
        rel_res = np.zeros_like(piv.values, dtype=float)
        for j in range(len(archetypes)):
            s_vec = piv.values[:, j]
            m_val = np.mean(s_vec)
            if m_val > 1.0:
                poly = np.polyfit(t_vals, s_vec, 2)
                rel_res[:, j] = (s_vec - np.polyval(poly, t_vals)) / m_val
        cov_raw = np.cov(rel_res, rowvar=False, ddof=1)
        cov_arch = 0.90 * cov_raw + 0.10 * np.diag(np.diag(cov_raw)) + 1e-6 * np.eye(len(archetypes))
        std_vec = np.sqrt(np.diag(cov_arch))
        corr_arch = cov_arch / np.outer(std_vec, std_vec)
        np.fill_diagonal(corr_arch, 1.0)

        df_mod = df_tr[df_tr['Year'] >= 2015]
        arch_mod = df_mod.groupby('Supplier_Archetype').agg(
            Value_USD=('Trade_Value_USD', 'sum'),
            Weight_Tonnes=('Weight_Tonnes', 'sum'),
            LWE_Tonnes=('LWE_Tonnes', 'sum')
        ).reindex(archetypes).reset_index()
        w_obs = (arch_mod['Weight_Tonnes'] / arch_mod['Weight_Tonnes'].sum()).values
        prices = (arch_mod['Value_USD'] / (arch_mod['Weight_Tonnes'] * 1000.0)).values
        lwe_vec = (arch_mod['LWE_Tonnes'] / arch_mod['Weight_Tonnes']).values
        haz_map = {
            'Intra-European Core': 1.597, 'Cold-Water Farmed Salmonids': 1.920,
            'North Atlantic & Arctic Capture': 0.267, 'Upwelling & Ocean Tuna Fisheries': 4.443,
            'Tropical Farmed Aquaculture': 5.302, 'Other Third Countries': 5.248,
            'Global Processing Hubs': 3.106, 'Americas Marine Capture': 1.814
        }
        haz_vec = np.array([haz_map[a] for a in archetypes])

        lb = np.array([0.58, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02])
        ub = np.array([0.74, 0.09, 0.08, 0.10, 0.12, 0.10, 0.09, 0.09])
        bounds = list(zip(lb, ub))
        cv_obs = float(np.sqrt(np.dot(w_obs, np.dot(cov_arch, w_obs))) * 100.0)
        cost_obs = float(np.dot(w_obs, prices))
        haz_obs = float(np.dot(w_obs, haz_vec))
        lwe_obs = float(np.dot(w_obs, lwe_vec))
        cons_pareto = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
            {'type': 'ineq', 'fun': lambda w: (cv_obs / 100.0) ** 2 - np.dot(w, np.dot(cov_arch, w))},
            {'type': 'ineq', 'fun': lambda w: haz_obs - np.dot(w, haz_vec)},
            {'type': 'ineq', 'fun': lambda w: cost_obs - np.dot(w, prices)},
            {'type': 'ineq', 'fun': lambda w: lwe_obs - np.dot(w, lwe_vec)}
        ]
        res_a0 = minimize(
            lambda w: 0.78 * (np.dot(w, haz_vec) / haz_obs) + 0.22 * (np.dot(w, prices) / cost_obs) + 0.04 * (np.dot(w, np.dot(cov_arch, w)) / ((cv_obs / 100.0) ** 2)),
            w_obs, method='SLSQP', bounds=bounds, constraints=cons_pareto, options={'maxiter': 600, 'ftol': 1e-12}
        )
        w_bio_opt = res_a0.x / np.sum(res_a0.x)
        cons_a1 = cons_pareto + [{'type': 'ineq', 'fun': lambda w: haz_obs * 0.985 - np.dot(w, haz_vec)}]
        res_a1 = minimize(
            lambda w: np.dot(w, np.dot(cov_arch, w)) / ((cv_obs / 100.0) ** 2) + 0.04 * (np.dot(w, haz_vec) / haz_obs),
            w_obs, method='SLSQP', bounds=bounds, constraints=cons_a1, options={'maxiter': 600, 'ftol': 1e-12}
        )
        w_vol_opt = res_a1.x / np.sum(res_a1.x)
        w_bal_arch = 0.50 * w_bio_opt + 0.50 * w_vol_opt

        arch_diag_df = pd.DataFrame({
            'Supplier_Archetype': archetypes,
            'Observed_Share_Pct': np.round(w_obs * 100.0, 2),
            'Biosecurity_Optimal_Share_Pct': np.round(w_bio_opt * 100.0, 2),
            'Balanced_Frontier_Share_Pct': np.round(w_bal_arch * 100.0, 2),
            'Minimum_Volatility_Share_Pct': np.round(w_vol_opt * 100.0, 2)
        })
        for idx_a, a_name in enumerate(archetypes):
            arch_diag_df[f'Corr_{idx_a+1}'] = np.round(corr_arch[:, idx_a], 3)
        arch_diag_df.to_csv(t8_path, index=False)

    return sp_params, df_bench, df_weights, df_pareto, df_arch, df_sov, df_shad, archetypes, corr_arch, arch_diag_df


# =============================================================================
# MAIN FIGURE 9: 6-PANEL (2x3) MULTI-OBJECTIVE PARETO & KKT SYNTHESIS PLATE
# =============================================================================
def generate_figure_9(sp_params, df_bench, df_weights, df_pareto, df_shad):
    print("\n[*] Generating Main Figure 9 (6-Panel Multi-Objective Pareto & KKT Synthesis Plate)...")
    fig = plt.figure(figsize=(23.0, 14.0), dpi=600)

    # Custom explicit axes coordinates providing dedicated margin for species labels
    ax_a = fig.add_axes([0.042, 0.555, 0.258, 0.405])
    ax_b = fig.add_axes([0.352, 0.555, 0.245, 0.405])
    ax_c = fig.add_axes([0.752, 0.555, 0.240, 0.405])

    ax_d = fig.add_axes([0.042, 0.062, 0.268, 0.405])
    ax_e = fig.add_axes([0.358, 0.062, 0.268, 0.405])
    ax_f = fig.add_axes([0.708, 0.062, 0.284, 0.405])

    # -------------------------------------------------------------------------
    # Panel (a): Pan-European Supply Volatility vs Wholesale Unit Cost Frontier
    # -------------------------------------------------------------------------
    p_pan = df_pareto[
        (df_pareto['Region'] == 'Pan-Europe (All 32 Nations)') &
        (df_pareto['Sweep_Type'] == 'Stability_vs_Cost')
    ].sort_values('Supply_Volatility_CV_Pct')

    b_pan = df_bench[df_bench['Region'] == 'Pan-Europe (All 32 Nations)'].copy()
    obs_row = b_pan[b_pan['Portfolio_Regime'].str.startswith('1.')].iloc[0]

    x_curve = p_pan['Supply_Volatility_CV_Pct'].values
    y_curve = p_pan['Wholesale_Unit_Cost_USD_KG'].values
    ax_a.fill_between(x_curve, y_curve, 5.75, color='#CBD5E1', alpha=0.24, zorder=1,
                      label='Dominated Sub-Optimal Region')
    ax_a.plot(x_curve, y_curve, color=OKABE_ITO['blue'], lw=2.9, zorder=3,
              label='Markowitz-Cost Efficient Frontier')

    ax_a.axvline(obs_row['Supply_Volatility_CV_Pct'], color='#94A3B8', ls='--', lw=1.1, zorder=2)
    ax_a.axhline(obs_row['Wholesale_Unit_Cost_USD_KG'], color='#94A3B8', ls='--', lw=1.1, zorder=2)

    regime_specs = [
        ('1.', 'Observed Baseline\n(11.95%, $5.51)', OKABE_ITO['vermilion'], 'X', 175, (10.15, 5.56)),
        ('2.', 'Minimum-Variance\n(8.38%, $5.30)', OKABE_ITO['blue'], 'D', 130, (8.68, 5.42)),
        ('3.', 'Minimum-Cost\n(11.44%, $4.26)', OKABE_ITO['orange'], 's', 130, (10.12, 4.38)),
        ('4.', 'Ecological and Biosecurity\n(11.26%, $4.93)', OKABE_ITO['purple'], '^', 145, (10.22, 5.15)),
        ('5.', 'Balanced Pareto-Optimal\n(9.02%, $4.94)', OKABE_ITO['green'], '*', 275, (8.12, 4.56)),
        ('6.', 'Planetary Health Shift\n(9.86%, $4.91)', OKABE_ITO['teal'], 'P', 160, (9.88, 4.62))
    ]

    for prefix, lbl_txt, col, mk, sz, (tx, ty) in regime_specs:
        r = b_pan[b_pan['Portfolio_Regime'].str.startswith(prefix)].iloc[0]
        xv, yv = r['Supply_Volatility_CV_Pct'], r['Wholesale_Unit_Cost_USD_KG']
        ax_a.scatter(xv, yv, c=col, marker=mk, s=sz, edgecolors='black', linewidths=1.1, zorder=6)
        ax_a.annotate(
            lbl_txt, xy=(xv, yv), xytext=(tx, ty),
            fontsize=9.6, fontweight='bold', color=col,
            bbox=dict(boxstyle='round,pad=0.20', facecolor='white', edgecolor=col, alpha=0.96, lw=0.95),
            arrowprops=dict(arrowstyle='->', color=col, lw=1.2, shrinkA=2, shrinkB=5),
            zorder=7
        )

    r5 = b_pan[b_pan['Portfolio_Regime'].str.startswith('5.')].iloc[0]
    ax_a.annotate(
        '', xy=(r5['Supply_Volatility_CV_Pct'] + 0.08, r5['Wholesale_Unit_Cost_USD_KG'] + 0.025),
        xytext=(obs_row['Supply_Volatility_CV_Pct'] - 0.08, obs_row['Wholesale_Unit_Cost_USD_KG'] - 0.025),
        arrowprops=dict(arrowstyle='-|>', color=OKABE_ITO['green'], lw=2.3, ls='-.'),
        zorder=5
    )

    ax_a.set_xlim(7.95, 12.48)
    ax_a.set_ylim(4.14, 5.72)
    ax_a.set_title("(a) Pan-European Supply Volatility and Wholesale Cost Frontier", fontweight='bold', loc='left', pad=8)
    ax_a.set_xlabel("Portfolio Supply Volatility (Coefficient of Variation, %)", fontweight='bold')
    ax_a.set_ylabel("Wholesale Unit Cost (US Dollars per Kilogram)", fontweight='bold')
    ax_a.yaxis.set_major_formatter(ticker.FormatStrFormatter('$%.2f'))
    ax_a.grid(True, linestyle=':', alpha=0.45)
    ax_a.legend(loc='lower left', frameon=True, facecolor='white', edgecolor='#CBD5E1', framealpha=0.96, fontsize=9.6)

    # -------------------------------------------------------------------------
    # Panel (b): Regional Biomass Conversion vs Food-Safety Hazard Trajectories
    # -------------------------------------------------------------------------
    reg_list = [
        ('Pan-Europe (All 32 Nations)', 'Pan-Europe (32 Nations)', OKABE_ITO['black'], (1.535, 2.04)),
        ('Western Europe', 'Western Europe', OKABE_ITO['blue'], (1.545, 2.52)),
        ('Southern Europe', 'Southern Europe', OKABE_ITO['vermilion'], (1.452, 2.45)),
        ('Central & Eastern Europe', 'Central and Eastern Europe', OKABE_ITO['orange'], (1.545, 1.66)),
        ('Northern Europe', 'Northern Europe', OKABE_ITO['green'], (1.548, 1.29))
    ]

    for reg_key, reg_disp, col, (tx, ty) in reg_list:
        sub = df_bench[df_bench['Region'] == reg_key]
        r_obs = sub[sub['Portfolio_Regime'].str.startswith('1.')].iloc[0]
        r_bal = sub[sub['Portfolio_Regime'].str.startswith('5.')].iloc[0]
        r_phd = sub[sub['Portfolio_Regime'].str.startswith('6.')].iloc[0]

        xo, yo = r_obs['LWE_Biomass_Multiplier'], r_obs['Hazard_Rate_per_100kt']
        xb, yb = r_bal['LWE_Biomass_Multiplier'], r_bal['Hazard_Rate_per_100kt']
        xp, yp = r_phd['LWE_Biomass_Multiplier'], r_phd['Hazard_Rate_per_100kt']

        ax_b.scatter(xo, yo, c=col, marker='o', s=115, edgecolors='black', lw=1.0, zorder=5)
        ax_b.scatter(xb, yb, c=col, marker='*', s=240, edgecolors='black', lw=1.0, zorder=6)
        ax_b.scatter(xp, yp, c=col, marker='D', s=95, edgecolors='black', lw=0.9, alpha=0.88, zorder=5)

        ax_b.annotate('', xy=(xb, yb), xytext=(xo, yo),
                      arrowprops=dict(arrowstyle='-|>', color=col, lw=2.0, shrinkA=4, shrinkB=5), zorder=4)
        ax_b.plot([xb, xp], [yb, yp], color=col, ls='--', lw=1.4, zorder=3)

        ax_b.annotate(
            reg_disp, xy=(xb, yb), xytext=(tx, ty),
            fontsize=9.6, fontweight='bold', color=col,
            bbox=dict(boxstyle='round,pad=0.18', facecolor='white', edgecolor=col, alpha=0.95, lw=0.85),
            arrowprops=dict(arrowstyle='->', color=col, lw=1.05, shrinkA=2, shrinkB=5),
            zorder=7
        )

    legend_b = [
        Line2D([0], [0], marker='o', color='w', label='Observed Baseline (2015–2024)', markerfacecolor='#475569', markeredgecolor='black', markersize=9),
        Line2D([0], [0], marker='*', color='w', label='Balanced Pareto-Optimal Portfolio', markerfacecolor='#475569', markeredgecolor='black', markersize=13),
        Line2D([0], [0], marker='D', color='w', label='Planetary Health Shift Portfolio', markerfacecolor='#475569', markeredgecolor='black', markersize=8)
    ]
    ax_b.set_xlim(1.44, 1.675)
    ax_b.set_ylim(1.20, 3.06)
    ax_b.set_title("(b) Regional Live-Weight Multiplier and Hazard Shifts", fontweight='bold', loc='left', pad=8)
    ax_b.set_xlabel("Live-Weight Equivalent Multiplier (Tonnes per Tonne)", fontweight='bold')
    ax_b.set_ylabel("Food-Safety Notifications per 100,000 Tonnes", fontweight='bold')
    ax_b.grid(True, linestyle=':', alpha=0.45)
    ax_b.legend(handles=legend_b, loc='upper left', frameon=True, facecolor='white', edgecolor='#CBD5E1', framealpha=0.96, fontsize=9.4)

    # -------------------------------------------------------------------------
    # Panel (c): Optimal 16-Species Portfolio Rebalancing Shifts (Full Names)
    # -------------------------------------------------------------------------
    w_bal_df = df_weights[
        (df_weights['Region'] == 'Pan-Europe (All 32 Nations)') &
        (df_weights['Portfolio_Regime'].str.startswith('5.'))
    ].copy()
    w_phd_df = df_weights[
        (df_weights['Region'] == 'Pan-Europe (All 32 Nations)') &
        (df_weights['Portfolio_Regime'].str.startswith('6.'))
    ].set_index('Species_Group')['Delta_vs_Observed_Pct_Pts']

    w_bal_df['PHD_Delta'] = w_bal_df['Species_Group'].map(w_phd_df)
    w_bal_df['Full_Label'] = w_bal_df['Species_Group'].map(FULL_SPECIES_LABELS)
    w_bal_df = w_bal_df.sort_values('Delta_vs_Observed_Pct_Pts', ascending=True).reset_index(drop=True)

    y_pos = np.arange(len(w_bal_df))
    deltas = w_bal_df['Delta_vs_Observed_Pct_Pts'].values
    phd_deltas = w_bal_df['PHD_Delta'].values
    bar_colors = [OKABE_ITO['green'] if v >= 0 else OKABE_ITO['vermilion'] for v in deltas]

    ax_c.barh(y_pos, deltas, height=0.66, color=bar_colors, edgecolor='black', lw=0.75, alpha=0.90)
    ax_c.scatter(phd_deltas, y_pos, color=OKABE_ITO['navy'], marker='D', s=46,
                 edgecolors='white', lw=0.7, zorder=6)
    ax_c.axvline(0, color='black', lw=1.1, zorder=4)

    for idx_y, (val, pval) in enumerate(zip(deltas, phd_deltas)):
        if val >= 0:
            right_anchor = max(val, pval) if pval > 0 else val
            ax_c.text(right_anchor + 0.35, idx_y, f"+{val:.2f}", va='center', ha='left',
                      fontsize=9.3, fontweight='bold', color='#065F46')
        else:
            left_anchor = min(val, pval) if pval < 0 else val
            ax_c.text(left_anchor - 0.35, idx_y, f"{val:.2f}", va='center', ha='right',
                      fontsize=9.3, fontweight='bold', color='#991B1B')

    legend_c = [
        Patch(facecolor=OKABE_ITO['green'], edgecolor='black', label='Expanding Allocation (Balanced Optimal)'),
        Patch(facecolor=OKABE_ITO['vermilion'], edgecolor='black', label='Moderating Allocation (Balanced Optimal)'),
        Line2D([0], [0], marker='D', color='w', label='Planetary Health Shift Allocation',
               markerfacecolor=OKABE_ITO['navy'], markeredgecolor='white', markersize=7.5)
    ]
    ax_c.set_yticks(y_pos)
    ax_c.set_yticklabels(w_bal_df['Full_Label'], fontsize=9.8, fontweight='bold')
    ax_c.set_xlim(-13.8, 14.8)
    ax_c.set_title("(c) Pan-European Sixteen-Species Allocation Shifts", fontweight='bold', loc='left', pad=8)
    ax_c.set_xlabel("Allocation Change Relative to Baseline (Percentage Points)", fontweight='bold')
    ax_c.grid(True, axis='x', linestyle=':', alpha=0.45)
    ax_c.legend(handles=legend_c, loc='lower right', frameon=True, facecolor='white', edgecolor='#CBD5E1', framealpha=0.96, fontsize=8.9)

    # -------------------------------------------------------------------------
    # Panel (d): Species Micronutrient Cost-Efficiency Frontier
    # -------------------------------------------------------------------------
    x_prot = sp_params['Protein_g_per_USD'].values
    y_epa = np.maximum(sp_params['EPA_DHA_mg_per_USD'].values, 25.0)
    b12_usd = sp_params['VitB12_ug_per_USD'].values
    tl_vals = sp_params['Calibrated_TL'].values
    sizes = np.clip(b12_usd * 28.0 + 65.0, 75.0, 450.0)

    sc_d = ax_d.scatter(
        x_prot, y_epa, s=sizes, c=tl_vals, cmap='viridis_r',
        vmin=2.0, vmax=4.5, edgecolors='black', linewidths=1.0, alpha=0.92, zorder=5
    )
    ax_d.set_yscale('log')
    cbar_d = fig.colorbar(sc_d, ax=ax_d, pad=0.018, fraction=0.045)
    cbar_d.set_label("Biological Trophic Level", fontweight='bold', fontsize=10.8)

    highlight_sp = {
        'Mackerel & Horse Mackerel': ('Mackerel and Horse Mackerel', (46.5, 11800.0)),
        'Herring, Sardines & Small Pelagics': ('Herring and Sardines', (44.0, 3600.0)),
        'Atlantic & Pacific Salmon': ('Atlantic Salmon', (6.0, 3200.0)),
        'Rainbow & Freshwater Trout': ('Rainbow Trout', (30.5, 2050.0)),
        'Bluefin Tuna & Major Tunas': ('Major Oceanic Tunas', (45.5, 1150.0)),
        'Mussels, Oysters & Bivalves': ('Mussels and Oysters', (3.5, 580.0)),
        'Cod, Hake, Pollock & Whitefish': ('Cod, Hake and Pollock', (41.0, 430.0)),
        'Pangasius, Tilapia & Freshwater Catfish': ('Pangasius and Tilapia', (41.0, 185.0)),
        'Shrimp & Prawns': ('Shrimp and Prawns', (19.0, 125.0)),
        'Crabs, Lobsters & Other Crustaceans': ('Crabs and Lobsters', (12.0, 72.0))
    }
    for _, row_s in sp_params.iterrows():
        sp_n = row_s['Species_Group']
        if sp_n in highlight_sp:
            lbl_s, (tx, ty) = highlight_sp[sp_n]
            px = row_s['Protein_g_per_USD']
            py = max(row_s['EPA_DHA_mg_per_USD'], 25.0)
            ax_d.annotate(
                lbl_s, xy=(px, py), xytext=(tx, ty),
                fontsize=9.2, fontweight='bold', color='#0F172A',
                bbox=dict(boxstyle='round,pad=0.16', facecolor='white', edgecolor='#94A3B8', alpha=0.94, lw=0.75),
                arrowprops=dict(arrowstyle='->', color='#475569', lw=0.95, shrinkA=2, shrinkB=4),
                zorder=6
            )

    ax_d.set_xlim(1.5, 92.0)
    ax_d.set_ylim(52.0, 19500.0)
    ax_d.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{int(y):,}"))
    ax_d.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax_d.set_title("(d) Species Micronutrient Delivery per US Dollar", fontweight='bold', loc='left', pad=8)
    ax_d.set_xlabel("Digestible Protein per US Dollar (Grams per Dollar)", fontweight='bold')
    ax_d.set_ylabel("Omega-3 Fatty Acids per Dollar (Milligrams, Log Scale)", fontweight='bold')
    ax_d.grid(True, which='both', linestyle=':', alpha=0.42)

    # -------------------------------------------------------------------------
    # Panel (e): Multi-Objective Percentage Gains Across Five Regimes (Ablation)
    # -------------------------------------------------------------------------
    r_obs_disp = obs_row['Extra_EU_Displaced_LWE_Factor']
    criteria_labels = [
        'Supply\nVolatility',
        'Wholesale\nUnit Cost',
        'Live-Weight\nMultiplier',
        'Displaced\nBiomass',
        'Food-Safety\nHazard Rate',
        'Daily Omega-3\nAvailability'
    ]
    regimes_compare = [
        ('2. Minimum-Variance (Markowitz Stability)', 'Minimum-Variance', OKABE_ITO['blue']),
        ('3. Minimum-Cost Nutrient-Adequate', 'Minimum-Cost', OKABE_ITO['orange']),
        ('4. Ecological & Biosecurity Boundary', 'Ecological and Biosecurity', OKABE_ITO['purple']),
        ('5. Balanced 4D Pareto-Optimal', 'Balanced Pareto-Optimal', OKABE_ITO['green']),
        ('6. Planetary Health & EFSA Omega-3 Shift', 'Planetary Health Shift', OKABE_ITO['teal'])
    ]

    x_e = np.arange(len(criteria_labels))
    bar_w = 0.155
    all_vals_matrix = np.zeros((len(regimes_compare), len(criteria_labels)))
    bal_bar_x = np.zeros(len(criteria_labels))

    for idx_r, (r_full, r_short, col) in enumerate(regimes_compare):
        rw = b_pan[b_pan['Portfolio_Regime'] == r_full].iloc[0]
        d_disp = (rw['Extra_EU_Displaced_LWE_Factor'] - r_obs_disp) / r_obs_disp * 100.0
        vals_r = [
            rw['Delta_CV_Rel_Pct'],
            rw['Delta_Cost_Rel_Pct'],
            rw['Delta_LWE_Rel_Pct'],
            d_disp,
            rw['Delta_Hazard_Rel_Pct'],
            rw['Delta_EPA_DHA_Rel_Pct']
        ]
        all_vals_matrix[idx_r, :] = vals_r
        offset = (idx_r - 2.0) * bar_w
        bars_e = ax_e.bar(x_e + offset, vals_r, width=bar_w, color=col, edgecolor='black', lw=0.65, label=r_short, zorder=4)
        if idx_r == 3:
            for c_i, bx in enumerate(bars_e):
                bal_bar_x[c_i] = bx.get_x() + bx.get_width() / 2.0

    # Place Balanced Pareto-Optimal callout cleanly outside group extrema with a thin leader line
    for c_i in range(len(criteria_labels)):
        v_bal = all_vals_matrix[3, c_i]
        if v_bal < 0:
            grp_min = np.min(all_vals_matrix[:, c_i])
            y_lbl = grp_min - 3.6
            ax_e.annotate(
                fmt_pct_1(v_bal, signed=True),
                xy=(bal_bar_x[c_i], v_bal),
                xytext=(x_e[c_i], y_lbl),
                ha='center', va='top', fontsize=8.9, fontweight='bold', color='#065F46',
                bbox=dict(boxstyle='round,pad=0.14', facecolor='#ECFDF5', edgecolor='#059669', lw=0.7),
                arrowprops=dict(arrowstyle='-', color='#059669', lw=0.9),
                zorder=7
            )
        else:
            grp_max = np.max(all_vals_matrix[:, c_i])
            y_lbl = grp_max + 2.8
            ax_e.annotate(
                fmt_pct_1(v_bal, signed=True),
                xy=(bal_bar_x[c_i], v_bal),
                xytext=(x_e[c_i], y_lbl),
                ha='center', va='bottom', fontsize=8.9, fontweight='bold', color='#065F46',
                bbox=dict(boxstyle='round,pad=0.14', facecolor='#ECFDF5', edgecolor='#059669', lw=0.7),
                arrowprops=dict(arrowstyle='-', color='#059669', lw=0.9),
                zorder=7
            )

    ax_e.axhline(0, color='black', lw=1.1, zorder=5)
    ax_e.set_xticks(x_e)
    ax_e.set_xticklabels(criteria_labels, fontsize=9.8, fontweight='bold')
    ax_e.set_ylim(-41.5, 43.0)
    ax_e.set_title("(e) Multi-Objective Percentage Changes Across Regimes", fontweight='bold', loc='left', pad=8)
    ax_e.set_ylabel("Change Relative to Observed Baseline (%)", fontweight='bold')
    ax_e.grid(True, axis='y', linestyle=':', alpha=0.45)
    ax_e.legend(loc='upper left', ncol=2, frameon=True, facecolor='white', edgecolor='#CBD5E1', framealpha=0.96, fontsize=9.0)

    # -------------------------------------------------------------------------
    # Panel (f): 4D Karush-Kuhn-Tucker Marginal Shadow Prices Across Regions
    # -------------------------------------------------------------------------
    sh_sub = df_shad[df_shad['Region'] != 'EU-27 Aggregate'].copy()
    region_full_map = {
        'Pan-Europe (All 32 Nations)': 'Pan-Europe\n(32 Nations)',
        'Western Europe': 'Western\nEurope',
        'Southern Europe': 'Southern\nEurope',
        'Central & Eastern Europe': 'Central and\nEastern Europe',
        'Northern Europe': 'Northern\nEurope'
    }
    sh_sub['Display_Region'] = sh_sub['Region'].map(region_full_map)
    y_f = np.arange(len(sh_sub))
    h_f = 0.195

    b1 = ax_f.barh(y_f - 1.5 * h_f, sh_sub['Annual_Capita_Shadow_Cost_Minus_1pct_CV_USD'], height=h_f,
                   color=OKABE_ITO['blue'], edgecolor='black', lw=0.65,
                   label='Supply Stability (-1% Volatility)')
    b2 = ax_f.barh(y_f - 0.5 * h_f, sh_sub['Annual_Capita_Shadow_Cost_Minus_005_LWE_USD'], height=h_f,
                   color=OKABE_ITO['green'], edgecolor='black', lw=0.65,
                   label='Ecological Footprint (-0.05 Live-Weight Multiplier)')
    b3 = ax_f.barh(y_f + 0.5 * h_f, sh_sub['Annual_Capita_Shadow_Cost_Minus_025_Hazard_USD'], height=h_f,
                   color=OKABE_ITO['orange'], edgecolor='black', lw=0.65,
                   label='Food-Safety Biosecurity (-0.25 Alerts per 100,000 Tonnes)')
    b4 = ax_f.barh(y_f + 1.5 * h_f, sh_sub['Annual_Capita_Shadow_Cost_Plus_50mg_EPA_USD'], height=h_f,
                   color=OKABE_ITO['purple'], edgecolor='black', lw=0.65,
                   label='Dietary Omega-3 Adequacy (+50 Milligrams per Day)')

    for bar_grp, t_col in [(b1, OKABE_ITO['blue']), (b2, '#065F46'), (b3, '#B45309'), (b4, '#86198F')]:
        for bar in bar_grp:
            wv = bar.get_width()
            ax_f.text(wv + 0.18, bar.get_y() + bar.get_height() / 2.0, f"${wv:.2f}",
                      va='center', ha='left', fontsize=8.9, fontweight='bold', color=t_col)

    ax_f.set_yticks(y_f)
    ax_f.set_yticklabels(sh_sub['Display_Region'], fontsize=10.2, fontweight='bold')
    ax_f.invert_yaxis()
    ax_f.set_xlim(0, 19.2)
    ax_f.xaxis.set_major_formatter(ticker.FormatStrFormatter('$%.0f'))
    ax_f.set_title("(f) Karush-Kuhn-Tucker Shadow Prices on Frontier", fontweight='bold', loc='left', pad=8)
    ax_f.set_xlabel("Marginal Shadow Cost (US Dollars per Capita per Year)", fontweight='bold')
    ax_f.grid(True, axis='x', linestyle=':', alpha=0.45)
    ax_f.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='#CBD5E1', framealpha=0.96, fontsize=8.9)

    save_dual_figure(fig, "Figure_9")


# =============================================================================
# SUPPLEMENTARY FIGURE S11: SUPPLIER ARCHETYPE FRONTIER & SHOCK CORRELATION
# =============================================================================
def generate_figure_s11(df_bench, df_arch, archetypes, corr_arch, arch_diag_df):
    print("\n[*] Generating Supplementary Figure S11 (Supplier Archetype Markowitz Frontier & Shock Correlation)...")
    fig = plt.figure(figsize=(20.2, 13.8), dpi=600)

    # Explicit coordinates guarantee clean space between Panel (a)'s colorbar and Panel (b)'s y-tick labels
    ax_a = fig.add_axes([0.048, 0.562, 0.395, 0.398])
    ax_b = fig.add_axes([0.605, 0.562, 0.358, 0.398])
    ax_c = fig.add_axes([0.145, 0.068, 0.335, 0.398])
    ax_d = fig.add_axes([0.555, 0.068, 0.425, 0.398])

    # -------------------------------------------------------------------------
    # Panel (a): Supplier Archetype Diversification & Biosecurity Frontier
    # -------------------------------------------------------------------------
    indiv = df_arch[df_arch['Record_Type'] == 'Individual_Archetype'].copy()
    obs_a = df_arch[df_arch['Record_Type'] == 'Observed_Portfolio'].iloc[0]
    front_a = df_arch[df_arch['Record_Type'] == 'Markowitz_Frontier'].sort_values('Portfolio_CV_Pct')

    sc_a = ax_a.scatter(
        indiv['Portfolio_CV_Pct'], indiv['Hazard_Rate_per_100kt'],
        s=indiv['Intra_EU_Share_Pct'] * 9.0 + 130.0,
        c=indiv['Unit_Cost_USD_KG'], cmap='plasma',
        edgecolors='black', linewidths=1.1, zorder=5
    )
    cbar_a = fig.colorbar(sc_a, ax=ax_a, pad=0.018, fraction=0.045)
    cbar_a.set_label("Wholesale Unit Cost (US Dollars per Kilogram)", fontweight='bold', fontsize=10.8)

    ax_a.plot(front_a['Portfolio_CV_Pct'], front_a['Hazard_Rate_per_100kt'],
              color=OKABE_ITO['green'], lw=3.4, zorder=6, label='Markowitz-Biosecurity Efficient Frontier')
    ax_a.scatter([front_a['Portfolio_CV_Pct'].iloc[0], front_a['Portfolio_CV_Pct'].iloc[-1]],
                 [front_a['Hazard_Rate_per_100kt'].iloc[0], front_a['Hazard_Rate_per_100kt'].iloc[-1]],
                 color=OKABE_ITO['green'], marker='*', s=140, edgecolors='black', lw=0.8, zorder=7)
    ax_a.scatter(obs_a['Portfolio_CV_Pct'], obs_a['Hazard_Rate_per_100kt'],
                 color=OKABE_ITO['vermilion'], marker='X', s=195, edgecolors='black', lw=1.1, zorder=8,
                 label='Observed Pan-European Sourcing (9.46%, 2.40)')

    ax_a.annotate(
        'Efficient Frontier\n(9.05%–9.46%, 2.17–2.37)',
        xy=(front_a['Portfolio_CV_Pct'].iloc[8], front_a['Hazard_Rate_per_100kt'].iloc[8]),
        xytext=(8.05, 3.30),
        fontsize=9.1, fontweight='bold', color='#065F46',
        bbox=dict(boxstyle='round,pad=0.16', facecolor='#ECFDF5', edgecolor='#059669', alpha=0.95, lw=0.8),
        arrowprops=dict(arrowstyle='->', color='#059669', lw=1.1, shrinkA=2, shrinkB=4),
        zorder=8
    )

    arch_offsets = {
        'Intra-European Core': (12.8, 0.92),
        'Cold-Water Farmed Salmonids': (20.2, 2.38),
        'North Atlantic & Arctic Capture': (17.2, 0.45),
        'Upwelling & Ocean Tuna Fisheries': (12.6, 4.10),
        'Tropical Farmed Aquaculture': (12.8, 5.58),
        'Other Third Countries': (16.4, 4.78),
        'Global Processing Hubs': (18.4, 3.55),
        'Americas Marine Capture': (17.5, 1.38)
    }
    for _, rw in indiv.iterrows():
        lbl_k = rw['Label']
        tx, ty = arch_offsets[lbl_k]
        ax_a.annotate(
            FULL_ARCHETYPE_LABELS[lbl_k],
            xy=(rw['Portfolio_CV_Pct'], rw['Hazard_Rate_per_100kt']),
            xytext=(tx, ty),
            fontsize=9.2, fontweight='bold', color='#0F172A',
            bbox=dict(boxstyle='round,pad=0.16', facecolor='white', edgecolor='#94A3B8', alpha=0.94, lw=0.75),
            arrowprops=dict(arrowstyle='->', color='#475569', lw=0.95, shrinkA=2, shrinkB=5),
            zorder=8
        )

    ax_a.set_xlim(7.8, 29.2)
    ax_a.set_ylim(-0.08, 6.18)
    ax_a.set_title("(a) Supplier Profile Volatility and Biosecurity Frontier", fontweight='bold', loc='left', pad=8)
    ax_a.set_xlabel("Supplier Profile Supply Volatility (Coefficient of Variation, %)", fontweight='bold')
    ax_a.set_ylabel("Food-Safety Notifications per 100,000 Tonnes", fontweight='bold')
    ax_a.grid(True, linestyle=':', alpha=0.45)
    ax_a.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='#CBD5E1', framealpha=0.96, fontsize=9.2)

    # -------------------------------------------------------------------------
    # Panel (b): Detrended Supply Shock Correlation Heatmap (8x8 Archetypes)
    # -------------------------------------------------------------------------
    im_b = ax_b.imshow(corr_arch, cmap='RdBu_r', vmin=-0.65, vmax=1.0, aspect='auto')
    cbar_b = fig.colorbar(im_b, ax=ax_b, pad=0.018, fraction=0.045)
    cbar_b.set_label("Detrended Supply Shock Correlation", fontweight='bold', fontsize=10.8)

    clean_arch_labels = [
        'Intra-European Core',
        'Cold-Water Salmonids',
        'Arctic Wild Capture',
        'Upwelling and Tuna',
        'Tropical Aquaculture',
        'Other Global Partners',
        'Global Reprocessing Hubs',
        'Americas Wild Capture'
    ]
    ax_b.set_xticks(np.arange(len(archetypes)))
    ax_b.set_yticks(np.arange(len(archetypes)))
    ax_b.set_xticklabels(clean_arch_labels, rotation=32, ha='right', fontsize=9.6, fontweight='bold')
    ax_b.set_yticklabels(clean_arch_labels, fontsize=9.8, fontweight='bold')

    for i in range(len(archetypes)):
        for j in range(len(archetypes)):
            val_c = corr_arch[i, j]
            txt_col = 'white' if abs(val_c) > 0.52 else '#0F172A'
            ax_b.text(j, i, f"{val_c:.2f}", ha='center', va='center', fontsize=9.2, fontweight='bold', color=txt_col)

    ax_b.set_title("(b) Interannual Supply Shock Correlation Matrix (1990–2024)", fontweight='bold', loc='left', pad=8)

    # -------------------------------------------------------------------------
    # Panel (c): Observed vs Optimal Supplier Archetype Allocation Shares
    # -------------------------------------------------------------------------
    y_c = np.arange(len(arch_diag_df))
    h_c = 0.25

    ax_c.barh(y_c - h_c, arch_diag_df['Observed_Share_Pct'], height=h_c, color=OKABE_ITO['slate'],
              edgecolor='black', lw=0.65, label='Observed Baseline Share (2015–2024)')
    ax_c.barh(y_c, arch_diag_df['Biosecurity_Optimal_Share_Pct'], height=h_c, color=OKABE_ITO['green'],
              edgecolor='black', lw=0.65, label='Biosecurity-Optimal Frontier Anchor')
    ax_c.barh(y_c + h_c, arch_diag_df['Minimum_Volatility_Share_Pct'], height=h_c, color=OKABE_ITO['blue'],
              edgecolor='black', lw=0.65, label='Minimum-Volatility Frontier Anchor')

    for idx_y, rw in arch_diag_df.iterrows():
        ax_c.text(rw['Observed_Share_Pct'] + 0.6, idx_y - h_c, f"{rw['Observed_Share_Pct']:.1f}%",
                  va='center', ha='left', fontsize=8.7, fontweight='bold', color=OKABE_ITO['slate'])
        ax_c.text(rw['Biosecurity_Optimal_Share_Pct'] + 0.6, idx_y, f"{rw['Biosecurity_Optimal_Share_Pct']:.1f}%",
                  va='center', ha='left', fontsize=8.7, fontweight='bold', color='#065F46')

    ax_c.set_yticks(y_c)
    ax_c.set_yticklabels([FULL_ARCHETYPE_LABELS[a] for a in arch_diag_df['Supplier_Archetype']], fontsize=9.8, fontweight='bold')
    ax_c.invert_yaxis()
    ax_c.set_xlim(0, 79.0)
    ax_c.set_title("(c) Observed and Efficient Supplier Profile Sourcing Shares", fontweight='bold', loc='left', pad=8)
    ax_c.set_xlabel("Share of Pan-European Seafood Import Volume (%)", fontweight='bold')
    ax_c.grid(True, axis='x', linestyle=':', alpha=0.45)
    ax_c.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='#CBD5E1', framealpha=0.96, fontsize=9.4)

    # -------------------------------------------------------------------------
    # Panel (d): Regional Extra-European Displaced Biomass & Omega-3 Adequacy
    # -------------------------------------------------------------------------
    regs_d = [
        'Pan-Europe (All 32 Nations)',
        'Western Europe',
        'Southern Europe',
        'Central & Eastern Europe',
        'Northern Europe'
    ]
    disp_names_d = [
        'Pan-Europe\n(32 Nations)',
        'Western\nEurope',
        'Southern\nEurope',
        'Central and\nEastern Europe',
        'Northern\nEurope'
    ]
    x_d = np.arange(len(regs_d))
    w_d = 0.25

    epa_obs_l, epa_bal_l, epa_phd_l = [], [], []
    disp_red_bal = []
    for rk in regs_d:
        sub = df_bench[df_bench['Region'] == rk]
        ro = sub[sub['Portfolio_Regime'].str.startswith('1.')].iloc[0]
        rb = sub[sub['Portfolio_Regime'].str.startswith('5.')].iloc[0]
        rp = sub[sub['Portfolio_Regime'].str.startswith('6.')].iloc[0]
        epa_obs_l.append(ro['Daily_EPA_DHA_mg_Capita'])
        epa_bal_l.append(rb['Daily_EPA_DHA_mg_Capita'])
        epa_phd_l.append(rp['Daily_EPA_DHA_mg_Capita'])
        disp_red_bal.append((rb['Extra_EU_Displaced_LWE_Factor'] - ro['Extra_EU_Displaced_LWE_Factor']) / ro['Extra_EU_Displaced_LWE_Factor'] * 100.0)

    ax_d.bar(x_d - w_d, epa_obs_l, width=w_d, color=OKABE_ITO['vermilion'], edgecolor='black', lw=0.7,
             label='Observed Baseline Omega-3 Availability')
    bars_bal_d = ax_d.bar(x_d, epa_bal_l, width=w_d, color=OKABE_ITO['green'], edgecolor='black', lw=0.7,
                          label='Balanced Pareto-Optimal Omega-3 Availability')
    ax_d.bar(x_d + w_d, epa_phd_l, width=w_d, color=OKABE_ITO['teal'], edgecolor='black', lw=0.7,
             label='Planetary Health Shift Omega-3 Availability')

    ax_d.axhline(250.0, color='#B91C1C', ls='--', lw=2.0, zorder=5,
                 label='European Food Safety Authority Target (250 Milligrams per Day)')

    for b_bar, d_red in zip(bars_bal_d, disp_red_bal):
        hv = b_bar.get_height()
        ax_d.text(
            b_bar.get_x() + b_bar.get_width() / 2.0, hv + 10.0,
            f"Displaced\n{d_red:.1f}%", ha='center', va='bottom',
            fontsize=8.6, fontweight='bold', color='#065F46',
            bbox=dict(boxstyle='round,pad=0.14', facecolor='#ECFDF5', edgecolor='#059669', lw=0.7)
        )

    ax_d.set_xticks(x_d)
    ax_d.set_xticklabels(disp_names_d, fontsize=10.0, fontweight='bold')
    ax_d.set_ylim(0, 615.0)
    ax_d.set_title("(d) Regional Omega-3 Adequacy and Displaced Biomass Cuts", fontweight='bold', loc='left', pad=8)
    ax_d.set_ylabel("Daily Omega-3 Fatty Acids (Milligrams per Capita)", fontweight='bold')
    ax_d.grid(True, axis='y', linestyle=':', alpha=0.45)
    ax_d.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='#CBD5E1', framealpha=0.96, fontsize=8.9)

    save_dual_figure(fig, "Figure_S11")


# =============================================================================
# SUPPLEMENTARY FIGURE S12: SOVEREIGN TWO-STAGE STABILIZATION & SENSITIVITY
# =============================================================================
def generate_figure_s12(df_sov, df_pareto):
    print("\n[*] Generating Supplementary Figure S12 (Sovereign Two-Stage Stabilization & Preference Sensitivity)...")
    fig = plt.figure(figsize=(19.5, 14.2), dpi=600)
    gs = gridspec.GridSpec(
        2, 2, figure=fig,
        left=0.075, right=0.988, top=0.962, bottom=0.062,
        wspace=0.255, hspace=0.265,
        height_ratios=[1.12, 0.95]
    )

    # -------------------------------------------------------------------------
    # Panel (a): Sovereign Two-Stage Volatility Stabilization Dumbbell (N=22 Core)
    # -------------------------------------------------------------------------
    ax_a = fig.add_subplot(gs[0, 0])
    sov_core = df_sov[
        (~df_sov['Is_ReExport_Hub']) &
        (~df_sov['Is_Microstate']) &
        (df_sov['Domestic_Catch_CV_Pct'].notna())
    ].sort_values('Domestic_Catch_CV_Pct', ascending=True).reset_index(drop=True)

    y_a = np.arange(len(sov_core))
    dom_cv = sov_core['Domestic_Catch_CV_Pct'].values
    obs_cv = sov_core['Observed_Supply_CV_Pct'].values
    opt_cv = sov_core['Optimal_Portfolio_Supply_CV_Pct'].values

    for i in range(len(sov_core)):
        c_min = min(dom_cv[i], obs_cv[i], opt_cv[i])
        c_max = max(dom_cv[i], obs_cv[i], opt_cv[i])
        ax_a.plot([c_min, c_max], [i, i], color='#CBD5E1', lw=2.2, zorder=2)
        ax_a.plot([obs_cv[i], opt_cv[i]], [i, i], color=OKABE_ITO['green'], lw=2.8, zorder=3)

    ax_a.scatter(dom_cv, y_a, color=OKABE_ITO['vermilion'], marker='o', s=75, edgecolors='black', lw=0.8, zorder=5,
                 label='Domestic Harvest Volatility (Autarky)')
    ax_a.scatter(obs_cv, y_a, color=OKABE_ITO['orange'], marker='s', s=75, edgecolors='black', lw=0.8, zorder=6,
                 label='Observed Trade-Stabilized Supply Volatility')
    ax_a.scatter(opt_cv, y_a, color=OKABE_ITO['green'], marker='*', s=145, edgecolors='black', lw=0.8, zorder=7,
                 label='Optimal Portfolio Supply Volatility')

    ax_a.set_yticks(y_a)
    ax_a.set_yticklabels(sov_core['Country_Name'], fontsize=9.6, fontweight='bold')
    ax_a.set_xlim(0, 108.0)
    ax_a.set_title("(a) Sovereign Two-Stage Volatility Stabilization (22 Consumer Nations)", fontweight='bold', loc='left', pad=8)
    ax_a.set_xlabel("Interannual Supply Volatility (Coefficient of Variation, %)", fontweight='bold')
    ax_a.grid(True, axis='x', linestyle=':', alpha=0.45)
    ax_a.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='#CBD5E1', framealpha=0.96, fontsize=9.3)

    # -------------------------------------------------------------------------
    # Panel (b): Sovereign No-Regret Multi-Objective Efficiency Gains (All 32 Nations)
    # -------------------------------------------------------------------------
    ax_b = fig.add_subplot(gs[0, 1])
    sov_all = df_sov.sort_values('Hazard_Reduction_Pct', ascending=True).reset_index(drop=True)
    y_b = np.arange(len(sov_all))

    ax_b.scatter(sov_all['Stage2_Optimization_Stabilization_Pct'], y_b, color=OKABE_ITO['blue'], marker='o', s=58,
                 edgecolors='black', lw=0.6, zorder=5, label='Stage-Two Volatility Reduction (%)')
    ax_b.scatter(sov_all['Cost_Efficiency_Saving_Pct'], y_b, color=OKABE_ITO['orange'], marker='s', s=55,
                 edgecolors='black', lw=0.6, zorder=5, label='Wholesale Unit Cost Saving (%)')
    ax_b.scatter(sov_all['LWE_Reduction_Pct'], y_b, color=OKABE_ITO['green'], marker='^', s=62,
                 edgecolors='black', lw=0.6, zorder=5, label='Live-Weight Multiplier Reduction (%)')
    ax_b.scatter(sov_all['Hazard_Reduction_Pct'], y_b, color=OKABE_ITO['vermilion'], marker='D', s=52,
                 edgecolors='black', lw=0.6, zorder=6, label='Food-Safety Hazard Reduction (%)')

    for i in range(len(sov_all)):
        min_v = min(
            sov_all.loc[i, 'Stage2_Optimization_Stabilization_Pct'],
            sov_all.loc[i, 'Cost_Efficiency_Saving_Pct'],
            sov_all.loc[i, 'LWE_Reduction_Pct'],
            sov_all.loc[i, 'Hazard_Reduction_Pct']
        )
        ax_b.plot([min_v, 0.0], [i, i], color='#E2E8F0', lw=1.3, zorder=2)

    ax_b.axvline(0.0, color='black', lw=1.2, ls='--', zorder=4)
    ax_b.set_yticks(y_b)
    ax_b.set_yticklabels(sov_all['Country_Name'], fontsize=8.8, fontweight='bold')
    ax_b.set_xlim(-46.5, 3.0)
    ax_b.set_title("(b) Sovereign No-Regret Efficiency Gains Across All 32 Nations", fontweight='bold', loc='left', pad=8)
    ax_b.set_xlabel("Change Relative to Observed National Baseline (%)", fontweight='bold')
    ax_b.grid(True, axis='x', linestyle=':', alpha=0.45)
    ax_b.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='#CBD5E1', framealpha=0.96, fontsize=8.8)

    # -------------------------------------------------------------------------
    # Panel (c): Macro-Regional Three-Stage Volatility Comparison
    # -------------------------------------------------------------------------
    ax_c = fig.add_subplot(gs[1, 0])
    macro_order = ['Central & Eastern Europe', 'Western Europe', 'Southern Europe', 'Northern Europe']
    macro_labels = ['Central and\nEastern Europe', 'Western\nEurope', 'Southern\nEurope', 'Northern\nEurope']
    m_dom, m_obs, m_opt = [], [], []
    for mr in macro_order:
        sub_m = sov_core[sov_core['Macro_Region'] == mr]
        if len(sub_m) == 0:
            sub_m = df_sov[df_sov['Macro_Region'] == mr]
        m_dom.append(sub_m['Domestic_Catch_CV_Pct'].mean())
        m_obs.append(sub_m['Observed_Supply_CV_Pct'].mean())
        m_opt.append(sub_m['Optimal_Portfolio_Supply_CV_Pct'].mean())

    x_c = np.arange(len(macro_order))
    w_c = 0.25
    b_dom = ax_c.bar(x_c - w_c, m_dom, width=w_c, color=OKABE_ITO['vermilion'], edgecolor='black', lw=0.75,
                     label='Stage Zero: Domestic Harvest Volatility')
    b_obs = ax_c.bar(x_c, m_obs, width=w_c, color=OKABE_ITO['orange'], edgecolor='black', lw=0.75,
                     label='Stage One: Observed Trade-Stabilized Supply')
    b_opt = ax_c.bar(x_c + w_c, m_opt, width=w_c, color=OKABE_ITO['green'], edgecolor='black', lw=0.75,
                     label='Stage Two: Optimal Portfolio Rebalancing')

    for b_grp, col_t in [(b_dom, '#991B1B'), (b_obs, '#B45309'), (b_opt, '#065F46')]:
        for br in b_grp:
            hv = br.get_height()
            ax_c.text(br.get_x() + br.get_width() / 2.0, hv + 0.7, f"{hv:.1f}%",
                      ha='center', va='bottom', fontsize=9.3, fontweight='bold', color=col_t)

    ax_c.set_xticks(x_c)
    ax_c.set_xticklabels(macro_labels, fontsize=10.2, fontweight='bold')
    ax_c.set_ylim(0, 42.5)
    ax_c.set_title("(c) Macro-Regional Three-Stage Volatility Decomposition", fontweight='bold', loc='left', pad=8)
    ax_c.set_ylabel("Mean Coefficient of Variation (%)", fontweight='bold')
    ax_c.grid(True, axis='y', linestyle=':', alpha=0.45)
    ax_c.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='#CBD5E1', framealpha=0.96, fontsize=9.4)

    # -------------------------------------------------------------------------
    # Panel (d): Continuous Preference-Weight Sensitivity Sweeps (Alpha in [0, 1])
    # -------------------------------------------------------------------------
    ax_d = fig.add_subplot(gs[1, 1])
    sw_stab = df_pareto[
        (df_pareto['Region'] == 'Pan-Europe (All 32 Nations)') &
        (df_pareto['Sweep_Type'] == 'Stability_vs_Cost')
    ].sort_values('Preference_Weight_Alpha')
    sw_eco = df_pareto[
        (df_pareto['Region'] == 'Pan-Europe (All 32 Nations)') &
        (df_pareto['Sweep_Type'] == 'Ecology_Hazard_vs_Cost')
    ].sort_values('Preference_Weight_Alpha')

    ax_d2 = ax_d.twinx()
    l1 = ax_d.plot(sw_stab['Preference_Weight_Alpha'], sw_stab['Supply_Volatility_CV_Pct'],
                   color=OKABE_ITO['blue'], marker='o', markersize=5.5, lw=2.3,
                   label='Supply Volatility (Stability Sweep, %)')
    l2 = ax_d.plot(sw_eco['Preference_Weight_Alpha'], sw_eco['Hazard_Rate_per_100kt'] * 4.0,
                   color=OKABE_ITO['purple'], marker='^', markersize=5.5, lw=2.1, ls='--',
                   label='Scaled Hazard Index (Ecology-Biosecurity Sweep)')
    l3 = ax_d2.plot(sw_stab['Preference_Weight_Alpha'], sw_stab['Wholesale_Unit_Cost_USD_KG'],
                    color=OKABE_ITO['orange'], marker='s', markersize=5.5, lw=2.3,
                    label='Wholesale Cost (Stability Sweep, US Dollars per Kilogram)')
    l4 = ax_d2.plot(sw_eco['Preference_Weight_Alpha'], sw_eco['Wholesale_Unit_Cost_USD_KG'],
                    color=OKABE_ITO['green'], marker='D', markersize=5.0, lw=2.0, ls='-.',
                    label='Wholesale Cost (Ecology Sweep, US Dollars per Kilogram)')

    ax_d.axvspan(0.40, 0.65, color=OKABE_ITO['green'], alpha=0.12)
    ax_d.text(0.525, 7.62, "Balanced Interior\nPolicy Zone", ha='center', va='bottom',
              fontsize=9.0, fontweight='bold', color='#065F46')

    ax_d.set_xlim(-0.02, 1.02)
    ax_d.set_ylim(7.45, 13.05)
    ax_d2.set_ylim(4.10, 5.75)
    ax_d2.yaxis.set_major_formatter(ticker.FormatStrFormatter('$%.2f'))

    ax_d.set_title("(d) Continuous Preference Weight Sensitivity Sweeps", fontweight='bold', loc='left', pad=8)
    ax_d.set_xlabel("Non-Monetary Policy Preference Weight (Alpha from 0.00 to 1.00)", fontweight='bold')
    ax_d.set_ylabel("Volatility (%) or Scaled Hazard Index", fontweight='bold', color=OKABE_ITO['blue'])
    ax_d2.set_ylabel("Wholesale Unit Cost (US Dollars per Kilogram)", fontweight='bold', color='#B45309')
    ax_d.grid(True, linestyle=':', alpha=0.45)

    lines_all = l1 + l2 + l3 + l4
    labels_all = [l.get_label() for l in lines_all]
    ax_d.legend(lines_all, labels_all, loc='upper center', ncol=2, frameon=True, facecolor='white',
                edgecolor='#CBD5E1', framealpha=0.96, fontsize=8.3)

    save_dual_figure(fig, "Figure_S12")


def main():
    print("=" * 80)
    print("PORTFOLIO OPTIMIZATION VISUALIZATION & TABLE SYNC ENGINE (600 DPI)")
    print("=" * 80)
    sp_params, df_bench, df_weights, df_pareto, df_arch, df_sov, df_shad, archetypes, corr_arch, arch_diag_df = (
        sync_tables_and_compute_archetype_diagnostics()
    )
    generate_figure_9(sp_params, df_bench, df_weights, df_pareto, df_shad)
    generate_figure_s11(df_bench, df_arch, archetypes, corr_arch, arch_diag_df)
    generate_figure_s12(df_sov, df_pareto)
    print("\n" + "=" * 80)
    print("COMPLETED ALL PORTFOLIO OPTIMIZATION FIGURES SUCCESSFULLY!")
    print("=" * 80)


if __name__ == '__main__':
    main()
