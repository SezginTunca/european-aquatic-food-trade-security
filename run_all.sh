#!/usr/bin/env bash
# ==============================================================================
# Replication Orchestrator: Continental Aquatic Food Trade Security
# Frontiers in Sustainable Food Systems (2026)
# Author: Sezgin Tunca (University of Southern Denmark)
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================================================"
echo "European Aquatic Food Trade Security — Replication Codebase"
echo "========================================================================"

MODE="${1:---all}"

run_figures() {
    echo ""
    echo ">>> Step 1/3: Running Markowitz-Pareto Portfolio Optimization (Script 12)..."
    python3 scripts/12_markowitz_pareto_portfolio_optimization.py

    echo ""
    echo ">>> Step 2/3: Generating 8 Advanced Main Publication Figures (Script 13)..."
    python3 scripts/13_generate_main_publication_figures.py

    echo ""
    echo ">>> Step 3/3: Generating 12 Supplementary Figures (Script 14)..."
    python3 scripts/14_generate_supplementary_figures.py

    echo ""
    echo "[✓] All publication figures and optimization tables generated in figures/ and tables/."
}

run_full_pipeline() {
    echo ""
    echo ">>> Executing full econometric and machine learning pipeline (Scripts 01-14)..."
    echo "NOTE: Script 01 requires raw UN Comtrade and FAO microdata in data/raw/."
    
    scripts=(
        "01_data_extraction_and_filtering.py"
        "02_compute_eu_econometric_indices.py"
        "03_network_centrality_and_modularity.py"
        "04_advanced_ml_gravity_and_simulations.py"
        "05_hazard_notifications.py"
        "06_consumer_access.py"
        "07_harvest_pressure.py"
        "08_nutrient_supply.py"
        "09_partner_matrix.py"
        "10_extra_european_gravity.py"
        "11_mirror_check.py"
        "12_markowitz_pareto_portfolio_optimization.py"
        "13_generate_main_publication_figures.py"
        "14_generate_supplementary_figures.py"
    )

    for s in "${scripts[@]}"; do
        echo ""
        echo ">>> Executing scripts/$s..."
        python3 "scripts/$s"
    done

    echo ""
    echo "[✓] Full pipeline execution completed successfully."
}

case "$MODE" in
    --figures|--figures-only)
        run_figures
        ;;
    --all|--pipeline)
        # Check if derived parquet exists, otherwise suggest running figures
        if [ ! -d "data/derived" ] || [ ! -f "data/derived/eu_species_bilateral_panel.parquet" ]; then
            echo ""
            echo "[INFO] Raw microdata / derived parquet files not detected in data/derived/."
            echo "[INFO] Running figures and portfolio optimizations using bundled analytical tables..."
            run_figures
        else
            run_full_pipeline
        fi
        ;;
    *)
        echo "Usage: ./run_all.sh [--figures | --all]"
        exit 1
        ;;
esac

