"""
GlowGuide: Comparative Analysis of Apriori and FP-Growth for Skincare Product Association Mining
Main Streamlit Application Entrypoint.
"""

from __future__ import annotations

import io
import os
import time
import pandas as pd
import streamlit as st

# Internal imports
from src.data_loader import (
    generate_synthetic_skincare_data,
    parse_and_preprocess_transactions,
    save_default_synthetic_dataset,
)
from src.mining import (
    encode_transactions_to_onehot,
    generate_rules,
    mine_frequent_itemsets,
    run_comparative_benchmark,
)
from src.routine_analysis import (
    detect_routine_column,
    run_routine_comparative_mining,
    segment_by_routine,
)
from src.theme import (
    apply_custom_theme,
    render_disclaimer,
    render_hero_banner,
    render_kpi,
)
from src.visualizations import (
    plot_basket_size_distribution,
    plot_benchmark_runtime,
    plot_cooccurrence_matrix,
    plot_routine_comparison_chart,
    plot_rules_scatter,
    plot_speedup_factor,
    plot_top_products,
    plot_top_rules_bar,
)


# Set page config
st.set_page_config(
    page_title="GlowGuide | Apriori vs FP-Growth Skincare Mining",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply skincare-inspired theme
apply_custom_theme()


# Session state initialization
if "dataset_source" not in st.session_state:
    st.session_state.dataset_source = "synthetic"
if "df_raw" not in st.session_state:
    synth_path = "data/synthetic_skincare_transactions.csv"
    if not os.path.exists(synth_path):
        save_default_synthetic_dataset(synth_path)
    st.session_state.df_raw = pd.read_csv(synth_path)
    st.session_state.is_synthetic = True
if "benchmark_results" not in st.session_state:
    st.session_state.benchmark_results = None


# Sidebar Navigation & Settings
with st.sidebar:
    st.markdown("## 🌿 GlowGuide")
    st.markdown("*Skincare Association Mining*")
    st.markdown("---")

    selected_page = st.radio(
        "Navigation",
        [
            "Home & Overview",
            "Dataset & Ingestion",
            "Algorithm Comparison",
            "Frequent Itemsets",
            "Association Rules",
            "AM / PM Routine Analysis",
            "Viva & Exam Guide",
        ],
        index=0,
    )

    st.markdown("---")
    st.markdown("### ⚙️ Quick Mining Parameters")
    sidebar_min_sup = st.slider(
        "Min Support (s)",
        min_value=0.01,
        max_value=0.30,
        value=0.05,
        step=0.01,
        help="Fraction of transactions that must contain the itemset.",
    )
    sidebar_min_conf = st.slider(
        "Min Confidence (c)",
        min_value=0.10,
        max_value=1.00,
        value=0.35,
        step=0.05,
        help="Conditional probability P(Consequent | Antecedent).",
    )
    sidebar_min_lift = st.slider(
        "Min Lift",
        min_value=1.0,
        max_value=4.0,
        value=1.1,
        step=0.1,
        help="Lift > 1 implies positive correlation between skincare items.",
    )

    st.markdown("---")
    st.caption("College Data Mining Demonstration Project • Python 3.13 & Streamlit")


# Data loading & preprocessing helper
def load_and_process_current_data(min_basket_size: int = 2):
    df_raw = st.session_state.df_raw
    transactions, df_meta, stats = parse_and_preprocess_transactions(
        df_raw,
        format_type="auto",
        min_basket_size=min_basket_size,
    )
    return transactions, df_meta, stats


# -------------------------------------------------------------------------------------------------
# PAGE 1: HOME & OVERVIEW
# -------------------------------------------------------------------------------------------------
if selected_page == "Home & Overview":
    render_hero_banner(
        title="GlowGuide",
        subtitle="Comparative Analysis of Apriori and FP-Growth for Skincare Product Association Mining",
        tag="Data Mining & Regimen Discovery",
    )

    render_disclaimer(is_synthetic=st.session_state.get("is_synthetic", True))

    col_intro, col_objectives = st.columns([1.1, 0.9])

    with col_intro:
        st.markdown("### 🌸 Project Motivation & Domain Context")
        st.write(
            """
            In modern retail and personal wellness, **skincare routines** are inherently sequential and associative.
            Consumers rarely purchase or apply isolated products; instead, they build **multi-step regimens**
            (Cleanser $\\rightarrow$ Active Serum $\\rightarrow$ Hydration $\\rightarrow$ SPF).

            Understanding **market basket co-occurrences** in skincare allows retailers to:
            - Architect synergistic e-commerce product recommendation bundles.
            - Optimize shelf placement and cart checkout cross-sell suggestions.
            - Understand customer segmentation based on routine preferences (e.g. morning photoprotection vs. nighttime cell renewal).
            """
        )

        st.markdown("### 🎯 Core Project Objectives")
        st.markdown(
            """
            1. **Frequent Itemset Mining:** Identify recurring skincare combinations using Apriori and FP-Growth.
            2. **Algorithmic Benchmarking:** Empirically evaluate execution runtime, memory footprint, and scalability across varying support thresholds.
            3. **Association Rule Extraction:** Derive high-confidence rules evaluated via **Support, Confidence, Lift, Leverage, and Jaccard Similarity**.
            4. **Chrono-Routine Segmentation:** Compare product affinity across **Morning (AM)** and **Evening (PM)** routines.
            5. **Educational Transparency:** Distinguish synthetic training samples from real surveys and enforce medical safety disclaimers.
            """
        )

    with col_objectives:
        st.markdown("### ⚖️ Theoretical Comparison: Apriori vs. FP-Growth")
        comparison_data = {
            "Dimension": [
                "Core Mining Philosophy",
                "Candidate Generation",
                "Database Passes",
                "Data Structure",
                "Time Complexity",
                "Space / Memory",
                "Bottleneck",
            ],
            "Apriori (Agrawal 1994)": [
                "Breadth-First (Level-wise)",
                "Combinatorial ($L_k \\to C_{k+1}$)",
                "$k_{max} + 1$ full scans",
                "Array / Hash-tree",
                "$O(2^{|Items|})$ worst case",
                "High (stores candidate sets)",
                "I/O scans & candidate explosion",
            ],
            "FP-Growth (Han 2000)": [
                "Depth-First (Divide & Conquer)",
                "Zero candidate generation",
                "Exactly 2 database scans",
                "Compact FP-Tree with node links",
                "$O(|Transactions| \\times |Items|)$",
                "Low to Moderate (in-memory tree)",
                "Tree building in memory for very sparse data",
            ],
        }
        st.table(pd.DataFrame(comparison_data))

    st.markdown("---")
    st.markdown("### 📐 Mathematical Formulation of Association Metrics")

    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        st.markdown("#### 1. Support")
        st.latex(r"\text{Support}(X \Rightarrow Y) = \frac{\sigma(X \cup Y)}{N} = P(X \cap Y)")
        st.caption("Proportion of transactions containing both itemsets $X$ and $Y$.")

        st.markdown("#### 4. Leverage")
        st.latex(r"\text{Leverage}(X \Rightarrow Y) = P(X \cap Y) - P(X)P(Y)")
        st.caption("Surplus frequency above expected co-occurrence under complete independence.")

    with m_col2:
        st.markdown("#### 2. Confidence")
        st.latex(r"\text{Confidence}(X \Rightarrow Y) = \frac{P(X \cap Y)}{P(X)} = P(Y \mid X)")
        st.caption("Conditional probability that basket contains $Y$ given it contains $X$.")

        st.markdown("#### 5. Conviction")
        st.latex(r"\text{Conviction}(X \Rightarrow Y) = \frac{1 - P(Y)}{1 - \text{Confidence}(X \Rightarrow Y)}")
        st.caption("Frequency ratio of rule error under independence vs actual observations.")

    with m_col3:
        st.markdown("#### 3. Lift")
        st.latex(r"\text{Lift}(X \Rightarrow Y) = \frac{P(X \cap Y)}{P(X) \cdot P(Y)}")
        st.caption("Strength of association. $\\text{Lift} > 1$ denotes positive correlation.")

        st.markdown("#### 6. Jaccard Similarity")
        st.latex(r"\text{Jaccard}(X, Y) = \frac{\sigma(X \cup Y)}{\sigma(X) + \sigma(Y) - \sigma(X \cup Y)}")
        st.caption("Intersection over union measuring symmetric affinity between skincare items.")


# -------------------------------------------------------------------------------------------------
# PAGE 2: DATASET & INGESTION
# -------------------------------------------------------------------------------------------------
elif selected_page == "Dataset & Ingestion":
    st.markdown("## 📂 Skincare Dataset & Preprocessing Pipeline")
    st.markdown("Upload your own skincare transaction CSV or explore the built-in synthetic benchmark dataset.")

    render_disclaimer(is_synthetic=st.session_state.get("is_synthetic", True))

    data_source_mode = st.radio(
        "Choose Data Source:",
        ["🌿 Built-in Synthetic Skincare Dataset (1,250 transactions)", "📤 Upload Custom CSV File"],
        index=0 if st.session_state.dataset_source == "synthetic" else 1,
        horizontal=True,
    )

    if data_source_mode.startswith("🌿"):
        st.session_state.dataset_source = "synthetic"
        synth_file = "data/synthetic_skincare_transactions.csv"
        if not os.path.exists(synth_file):
            save_default_synthetic_dataset(synth_file)
        st.session_state.df_raw = pd.read_csv(synth_file)
        st.session_state.is_synthetic = True

        st.info("Currently using the **Synthetic Skincare Demo Dataset** with pre-configured morning/evening routines and realistic dermatological itemsets.")

        with st.expander("🔄 Re-generate Fresh Synthetic Dataset with Custom Size"):
            gen_col1, gen_col2 = st.columns([1, 1])
            with gen_col1:
                custom_n = st.slider("Number of synthetic transactions", 300, 3000, 1250, step=50)
                custom_seed = st.number_input("Random Seed", value=42, step=1)
            with gen_col2:
                if st.button("Generate & Load New Synthetic Dataset"):
                    new_df = generate_synthetic_skincare_data(num_transactions=custom_n, seed=int(custom_seed))
                    new_df.to_csv("data/synthetic_skincare_transactions.csv", index=False)
                    st.session_state.df_raw = new_df
                    st.success(f"Generated {custom_n} fresh synthetic skincare transactions!")
                    st.rerun()

    else:
        st.session_state.dataset_source = "custom"
        st.session_state.is_synthetic = False
        uploaded_file = st.file_uploader(
            "Upload Skincare Transactions CSV",
            type=["csv"],
            help="Supports basket strings (e.g. 'Cleanser, Vitamin C, SPF'), long transaction-item pairs, or one-hot matrices.",
        )

        if uploaded_file is not None:
            try:
                uploaded_df = pd.read_csv(uploaded_file)
                if uploaded_df.empty:
                    st.error("Uploaded CSV file is empty. Please provide a valid CSV.")
                else:
                    st.session_state.df_raw = uploaded_df
                    st.success(f"Successfully loaded {uploaded_file.name} with {len(uploaded_df)} rows!")
            except Exception as e:
                st.error(f"Error parsing uploaded file: {str(e)}")

    st.markdown("---")
    st.markdown("### 🧹 Data Cleaning & Preprocessing Settings")

    prep_c1, prep_c2 = st.columns([1, 1])
    with prep_c1:
        min_basket_filter = st.slider(
            "Minimum Basket Size Filter",
            min_value=1,
            max_value=5,
            value=2,
            help="Association rules require at least 2 items per transaction.",
        )
    with prep_c2:
        st.write("**Cleaning Pipeline Steps:**")
        st.markdown(
            """
            - Trim leading/trailing whitespace and title-case product names
            - Deduplicate items within the same basket
            - Filter out empty records and null entries
            - Exclude baskets below minimum size threshold
            """
        )

    # Process transactions
    transactions, df_meta, stats = load_and_process_current_data(min_basket_size=min_basket_filter)

    if not transactions:
        st.warning("No valid transactions found after applying preprocessing filters. Lower the minimum basket size or verify your CSV.")
    else:
        # Display Stats KPI Row
        st.markdown("### 📊 Ingestion & Preprocessing Summary")
        k1, k2, k3, k4, k5 = st.columns(5)
        with k1:
            render_kpi("Raw Records", f"{stats['raw_record_count']:,}")
        with k2:
            render_kpi("Clean Baskets", f"{stats['valid_transactions_count']:,}", f"Dropped {stats['filtered_out_count']}")
        with k3:
            render_kpi("Unique Items", f"{stats['unique_items_count']:,}")
        with k4:
            render_kpi("Avg Basket Size", f"{stats['mean_basket_size']:.2f}")
        with k5:
            render_kpi("Max Basket Size", f"{stats['max_basket_size']}")

        # Visualizations
        tab_view, tab_charts = st.tabs(["📋 Data Preview & Tables", "📈 Exploratory Charts"])

        with tab_view:
            st.markdown("#### Cleaned Transactions Preview (First 10 records)")
            preview_rows = []
            for i, t in enumerate(transactions[:10], 1):
                preview_rows.append({
                    "Transaction #": i,
                    "Basket Size": len(t),
                    "Skincare Products in Basket": ", ".join(t),
                })
            st.dataframe(pd.DataFrame(preview_rows), use_container_width=True)

            if not df_meta.empty:
                st.markdown("#### Metadata Attributes Preview")
                st.dataframe(df_meta.head(10), use_container_width=True)

        with tab_charts:
            chart_col1, chart_col2 = st.columns(2)
            with chart_col1:
                st.plotly_chart(plot_top_products(transactions, top_n=12), use_container_width=True)
            with chart_col2:
                st.plotly_chart(plot_basket_size_distribution(transactions), use_container_width=True)

            st.plotly_chart(plot_cooccurrence_matrix(transactions, top_n=10), use_container_width=True)


# -------------------------------------------------------------------------------------------------
# PAGE 3: ALGORITHM COMPARISON (APRIORI VS FP-GROWTH)
# -------------------------------------------------------------------------------------------------
elif selected_page == "Algorithm Comparison":
    st.markdown("## ⚡ Comparative Benchmark: Apriori vs. FP-Growth")
    st.markdown("Evaluate execution time, candidate generation overhead, and scaling behavior across support thresholds.")

    render_disclaimer(is_synthetic=st.session_state.get("is_synthetic", True))

    transactions, df_meta, stats = load_and_process_current_data()
    if not transactions:
        st.error("No valid transactions available. Please configure the dataset first.")
        st.stop()

    df_onehot, _ = encode_transactions_to_onehot(transactions)

    st.markdown("### 🧪 Live Side-by-Side Execution Test")
    col_bench_ctrl1, col_bench_ctrl2 = st.columns([1, 1])

    with col_bench_ctrl1:
        test_support = st.slider(
            "Test Minimum Support Threshold",
            min_value=0.01,
            max_value=0.25,
            value=0.05,
            step=0.01,
            key="test_support_slider",
        )
    with col_bench_ctrl2:
        test_max_len = st.selectbox(
            "Max Itemset Length (Optional constraint)",
            [None, 2, 3, 4],
            index=0,
            format_func=lambda x: "No Limit (Full Lattice)" if x is None else f"Max {x} Items",
        )

    if st.button("🚀 Run Live Comparative Test", type="primary"):
        with st.spinner("Executing Apriori and FP-Growth on transactions..."):
            # Run Apriori
            ap_items, ap_time = mine_frequent_itemsets(
                df_onehot, algorithm="apriori", min_support=test_support, max_len=test_max_len
            )
            # Run FP-Growth
            fp_items, fp_time = mine_frequent_itemsets(
                df_onehot, algorithm="fpgrowth", min_support=test_support, max_len=test_max_len
            )

        st.markdown("#### ⏱️ Real-Time Performance Results")
        res_c1, res_c2, res_c3, res_c4 = st.columns(4)

        speedup = (ap_time / fp_time) if fp_time > 0 else 1.0

        with res_c1:
            render_kpi("Apriori Runtime", f"{ap_time * 1000:.2f} ms", f"{len(ap_items)} itemsets")
        with res_c2:
            render_kpi("FP-Growth Runtime", f"{fp_time * 1000:.2f} ms", f"{len(fp_items)} itemsets")
        with res_c3:
            render_kpi("Speedup Factor", f"{speedup:.1f}x", "FP-Growth vs Apriori")
        with res_c4:
            match_status = "✅ 100% Identical" if len(ap_items) == len(fp_items) else "⚠️ Discrepancy"
            render_kpi("Mathematical Equivalence", match_status, "Itemset Count Match")

        # Explain why
        with st.expander("🔍 Why did FP-Growth perform faster?"):
            st.markdown(
                f"""
                - **Apriori Runtime ({ap_time*1000:.2f} ms):** Apriori is a breadth-first level-wise algorithm.
                  It repeatedly scans the database to test candidate itemsets $C_k$ derived from $L_{{k-1}}$.
                  As the minimum support decreases, the number of candidate pairs grows combinatorially ($O(2^d)$).
                - **FP-Growth Runtime ({fp_time*1000:.2f} ms):** FP-Growth compresses transactions into an in-memory
                  **FP-Tree** requiring only **2 database scans**. Frequent itemsets are mined using conditional
                  FP-trees and prefix paths without generating any intermediate candidate sets.
                """
            )

    st.markdown("---")
    st.markdown("### 📈 Multi-Threshold Benchmark Suite")
    st.write(
        "Run an automated sweep across multiple support thresholds to visualize how Apriori runtime diverges exponentially from FP-Growth."
    )

    if st.button("📊 Run Multi-Threshold Benchmark Sweep"):
        with st.spinner("Benchmarking across 6 support thresholds (3 repeats per test)..."):
            bench_df, insights = run_comparative_benchmark(
                df_onehot,
                support_thresholds=[0.03, 0.05, 0.08, 0.12, 0.18, 0.25],
                num_runs=3,
            )
            st.session_state.benchmark_results = bench_df

    if st.session_state.benchmark_results is not None:
        b_df = st.session_state.benchmark_results

        b_c1, b_c2 = st.columns(2)
        with b_c1:
            st.plotly_chart(plot_benchmark_runtime(b_df), use_container_width=True)
        with b_c2:
            st.plotly_chart(plot_speedup_factor(b_df), use_container_width=True)

        st.markdown("#### Benchmark Metric Table")
        display_bench = b_df[[
            "min_support_pct",
            "itemset_count",
            "apriori_time_ms",
            "fpgrowth_time_ms",
            "fp_speedup_factor",
            "counts_match",
        ]].rename(columns={
            "min_support_pct": "Min Support",
            "itemset_count": "Frequent Itemsets",
            "apriori_time_ms": "Apriori (ms)",
            "fpgrowth_time_ms": "FP-Growth (ms)",
            "fp_speedup_factor": "FP Speedup Multiplier",
            "counts_match": "Itemsets Match",
        })
        st.dataframe(display_bench, use_container_width=True)

        # Download benchmark CSV
        csv_buffer = io.StringIO()
        b_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Download Benchmark Results (CSV)",
            data=csv_buffer.getvalue(),
            file_name="glowguide_apriori_vs_fpgrowth_benchmark.csv",
            mime="text/csv",
        )


# -------------------------------------------------------------------------------------------------
# PAGE 4: FREQUENT ITEMSETS
# -------------------------------------------------------------------------------------------------
elif selected_page == "Frequent Itemsets":
    st.markdown("## 🔍 Frequent Itemsets Explorer")
    st.markdown("Inspect all combinations of skincare products that meet the minimum support threshold.")

    render_disclaimer(is_synthetic=st.session_state.get("is_synthetic", True))

    transactions, _, stats = load_and_process_current_data()
    if not transactions:
        st.error("No valid transactions loaded.")
        st.stop()

    df_onehot, _ = encode_transactions_to_onehot(transactions)

    # Algorithm and threshold selection
    col_algo, col_sup, col_len = st.columns([1, 1, 1])
    with col_algo:
        algo_choice = st.selectbox("Mining Algorithm", ["FP-Growth", "Apriori"], index=0)
    with col_sup:
        cur_sup = st.slider("Minimum Support Threshold", 0.01, 0.30, sidebar_min_sup, 0.01)
    with col_len:
        max_len_choice = st.selectbox(
            "Max Itemset Size",
            [None, 1, 2, 3, 4, 5],
            index=0,
            format_func=lambda x: "All Sizes" if x is None else f"{x} items",
        )

    itemsets, exec_time = mine_frequent_itemsets(
        df_onehot,
        algorithm="fpgrowth" if algo_choice == "FP-Growth" else "apriori",
        min_support=cur_sup,
        max_len=max_len_choice,
    )

    if itemsets.empty:
        st.warning(f"No itemsets found with minimum support ≥ {cur_sup*100:.1f}%. Try lowering the threshold.")
    else:
        st.markdown(
            f"Found **{len(itemsets)}** frequent itemsets using **{algo_choice}** in **{exec_time*1000:.2f} ms**."
        )

        # Clean display DataFrame
        itemsets_display = itemsets.copy()
        itemsets_display["Products in Itemset"] = itemsets_display["itemsets"].apply(
            lambda s: ", ".join(sorted(list(s)))
        )
        itemsets_display["Support %"] = (itemsets_display["support"] * 100).round(2)
        itemsets_display["Itemset Size"] = itemsets_display["length"]

        # Search filter
        search_kw = st.text_input("Filter by skincare ingredient / product name:", "")
        if search_kw.strip():
            itemsets_display = itemsets_display[
                itemsets_display["Products in Itemset"].str.contains(search_kw.strip(), case=False)
            ]

        # Table display
        st.dataframe(
            itemsets_display[["Itemset Size", "Support %", "Products in Itemset"]],
            use_container_width=True,
            height=450,
        )

        # Download button
        csv_itemsets = io.StringIO()
        itemsets_display.to_csv(csv_itemsets, index=False)
        st.download_button(
            label="📥 Download Frequent Itemsets (CSV)",
            data=csv_itemsets.getvalue(),
            file_name=f"glowguide_frequent_itemsets_{algo_choice.lower()}.csv",
            mime="text/csv",
        )


# -------------------------------------------------------------------------------------------------
# PAGE 5: ASSOCIATION RULES
# -------------------------------------------------------------------------------------------------
elif selected_page == "Association Rules":
    st.markdown("## 💡 Skincare Association Rules & Basket Affinity")
    st.markdown("Discover actionable co-purchase and routine rules with confidence, lift, leverage, and Jaccard metrics.")

    render_disclaimer(is_synthetic=st.session_state.get("is_synthetic", True))

    transactions, _, stats = load_and_process_current_data()
    if not transactions:
        st.error("No valid transactions loaded.")
        st.stop()

    df_onehot, _ = encode_transactions_to_onehot(transactions)

    # Mining rules
    with st.spinner("Mining frequent itemsets & deriving association rules..."):
        frequent_itemsets, _ = mine_frequent_itemsets(
            df_onehot, algorithm="fpgrowth", min_support=sidebar_min_sup
        )
        rules = generate_rules(
            frequent_itemsets,
            metric="confidence",
            min_threshold=sidebar_min_conf,
            min_lift=sidebar_min_lift,
        )

    if rules.empty:
        st.warning(
            f"No association rules found with Support ≥ {sidebar_min_sup*100:.1f}%, "
            f"Confidence ≥ {sidebar_min_conf*100:.1f}%, and Lift ≥ {sidebar_min_lift:.1f}. "
            "Try loosening the sidebar threshold sliders."
        )
    else:
        # Rule KPIs
        rk1, rk2, rk3, rk4 = st.columns(4)
        with rk1:
            render_kpi("Rules Discovered", f"{len(rules):,}")
        with rk2:
            render_kpi("Max Lift", f"{rules['lift'].max():.2f}", "Strongest Correlation")
        with rk3:
            render_kpi("Max Confidence", f"{rules['confidence'].max() * 100:.1f}%")
        with rk4:
            render_kpi("Avg Jaccard Index", f"{rules['jaccard'].mean():.3f}")

        # Visualizations
        tab_scatter, tab_bar, tab_table, tab_explain = st.tabs([
            "📊 Support vs. Confidence Landscape",
            "🏆 Top Rules by Lift / Metric",
            "📋 Comprehensive Rules Table",
            "🗣️ Plain-English Rule Translator",
        ])

        with tab_scatter:
            st.plotly_chart(plot_rules_scatter(rules), use_container_width=True)

        with tab_bar:
            metric_pick = st.selectbox(
                "Rank Top Rules By:",
                ["lift", "confidence", "support", "jaccard", "leverage"],
                index=0,
            )
            st.plotly_chart(plot_top_rules_bar(rules, metric=metric_pick, top_n=10), use_container_width=True)

        with tab_table:
            # Interactive filter by ingredient
            rule_filter = st.text_input("Filter rules containing product:", "", key="rule_search")
            filtered_rules = rules.copy()
            if rule_filter.strip():
                filtered_rules = filtered_rules[
                    filtered_rules["rule_str"].str.contains(rule_filter.strip(), case=False)
                ]

            # Format table for clean display
            display_cols = [
                "antecedents_str",
                "consequents_str",
                "support",
                "confidence",
                "lift",
                "leverage",
                "jaccard",
            ]
            rules_show = filtered_rules[display_cols].copy()
            rules_show["support"] = (rules_show["support"] * 100).round(2).astype(str) + "%"
            rules_show["confidence"] = (rules_show["confidence"] * 100).round(1).astype(str) + "%"
            rules_show["lift"] = rules_show["lift"].round(2)
            rules_show["leverage"] = rules_show["leverage"].round(4)
            rules_show["jaccard"] = rules_show["jaccard"].round(3)

            rules_show = rules_show.rename(columns={
                "antecedents_str": "Antecedent (IF)",
                "consequents_str": "Consequent (THEN)",
                "support": "Support",
                "confidence": "Confidence",
                "lift": "Lift",
                "leverage": "Leverage",
                "jaccard": "Jaccard",
            })

            st.dataframe(rules_show, use_container_width=True, height=450)

            # Download CSV
            csv_rules = io.StringIO()
            filtered_rules.to_csv(csv_rules, index=False)
            st.download_button(
                label="📥 Download Association Rules (CSV)",
                data=csv_rules.getvalue(),
                file_name="glowguide_association_rules.csv",
                mime="text/csv",
            )

        with tab_explain:
            st.markdown("#### 💬 Plain-English Business & Consumer Interpretation")
            st.write("Select any discovered rule below to see how a cosmetic brand manager or data analyst would interpret it.")

            selected_rule_idx = st.selectbox(
                "Choose a rule to translate:",
                range(len(rules)),
                format_func=lambda i: rules.iloc[i]["rule_str"],
            )

            chosen = rules.iloc[selected_rule_idx]
            ant = chosen["antecedents_str"]
            con = chosen["consequents_str"]
            sup_pct = chosen["support"] * 100
            conf_pct = chosen["confidence"] * 100
            lift_val = chosen["lift"]
            jacc_val = chosen["jaccard"]

            st.markdown(
                f"""
                <div class="glow-card">
                    <div class="glow-card-title">📖 Rule Translation: {{{ant}}} → {{{con}}}</div>
                    <p><strong>Retail & Consumer Insight:</strong></p>
                    <ul>
                        <li><strong>Occurrence:</strong> This combination appears in <strong>{sup_pct:.1f}%</strong> of all skincare baskets.</li>
                        <li><strong>Likelihood:</strong> When a shopper selects <em>{ant}</em>, there is a <strong>{conf_pct:.1f}% conditional probability</strong> that they will also add <em>{con}</em>.</li>
                        <li><strong>Affinity (Lift):</strong> Customers purchasing <em>{ant}</em> are <strong>{lift_val:.2f} times more likely</strong> to purchase <em>{con}</em> compared to the general baseline rate of <em>{con}</em>.</li>
                        <li><strong>Symmetric Jaccard Affinity:</strong> {jacc_val:.3f} — indicating strong mutual routine bundling.</li>
                    </ul>
                    <p style="color: #4A6B5B; font-weight: 600;">Recommended Action: Package these two products as a curated morning/evening duo or trigger an e-commerce checkout cross-sell modal.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


# -------------------------------------------------------------------------------------------------
# PAGE 6: AM / PM ROUTINE ANALYSIS
# -------------------------------------------------------------------------------------------------
elif selected_page == "AM / PM Routine Analysis":
    st.markdown("## ☀️🌙 Morning (AM) vs. Evening (PM) Routine Segmentation")
    st.markdown("Contrast daytime environmental defense patterns against nighttime cellular regeneration routines.")

    render_disclaimer(is_synthetic=st.session_state.get("is_synthetic", True))

    transactions, df_meta, _ = load_and_process_current_data()
    routine_col = detect_routine_column(df_meta)

    if not routine_col:
        st.warning(
            "No routine column (e.g. 'routine', 'time_of_day') was detected in the active dataset. "
            "Switch to the **Synthetic Skincare Demo Dataset** in the 'Dataset & Ingestion' page to explore routine segmentation."
        )
    else:
        st.success(f"Detected routine column: **'{routine_col}'**.")

        # Segment transactions
        segments = segment_by_routine(transactions, df_meta, routine_col=routine_col)

        # Run comparative mining for each segment
        with st.spinner("Mining segmented AM and PM routines..."):
            routine_results = run_routine_comparative_mining(
                segments,
                min_support=0.06,
                min_confidence=0.35,
            )

        # Summary chart
        st.plotly_chart(plot_routine_comparison_chart(routine_results), use_container_width=True)

        st.markdown("---")
        st.markdown("### 🧴 Side-by-Side Routine Comparison")

        am_key = "Morning (AM)"
        pm_key = "Evening (PM)"

        col_am, col_pm = st.columns(2)

        with col_am:
            st.markdown(
                """
                <div style="background: #FFFDF9; border: 1px solid #FDE68A; border-radius: 12px; padding: 16px; margin-bottom: 12px;">
                    <span class="routine-am-badge">☀️ Morning (AM) Regimen</span>
                    <h4 style="margin: 8px 0 4px 0; color: #92400E;">Daytime Defense & UV Protection</h4>
                    <p style="font-size: 0.88rem; color: #78350F;">
                        Morning routines prioritize <strong>antioxidant photoprotection</strong> (Vitamin C),
                        lightweight non-comedogenic hydration, and <strong>broad-spectrum UV filters (SPF 50)</strong>.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if am_key in routine_results:
                am_data = routine_results[am_key]
                st.write(f"**Transactions:** {am_data['transaction_count']} | **Rules:** {len(am_data['rules'])}")

                if not am_data["rules"].empty:
                    st.markdown("**Top Morning Association Rules:**")
                    am_rules_show = am_data["rules"].head(5)[[
                        "rule_str", "confidence", "lift"
                    ]].copy()
                    am_rules_show["confidence"] = (am_rules_show["confidence"] * 100).round(1).astype(str) + "%"
                    am_rules_show["lift"] = am_rules_show["lift"].round(2)
                    st.dataframe(am_rules_show, use_container_width=True)
                else:
                    st.info("No morning rules found above threshold.")

        with col_pm:
            st.markdown(
                """
                <div style="background: #F8FAFC; border: 1px solid #C7D2FE; border-radius: 12px; padding: 16px; margin-bottom: 12px;">
                    <span class="routine-pm-badge">🌙 Evening (PM) Regimen</span>
                    <h4 style="margin: 8px 0 4px 0; color: #3730A3;">Nighttime Repair & Cell Renewal</h4>
                    <p style="font-size: 0.88rem; color: #312E81;">
                        Evening routines center around <strong>double cleansing</strong> (Balm + Cleanser),
                        photosensitive cell-turnover actives (<strong>Retinoids</strong>, AHA/BHA), and barrier replenishing ceramides.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if pm_key in routine_results:
                pm_data = routine_results[pm_key]
                st.write(f"**Transactions:** {pm_data['transaction_count']} | **Rules:** {len(pm_data['rules'])}")

                if not pm_data["rules"].empty:
                    st.markdown("**Top Evening Association Rules:**")
                    pm_rules_show = pm_data["rules"].head(5)[[
                        "rule_str", "confidence", "lift"
                    ]].copy()
                    pm_rules_show["confidence"] = (pm_rules_show["confidence"] * 100).round(1).astype(str) + "%"
                    pm_rules_show["lift"] = pm_rules_show["lift"].round(2)
                    st.dataframe(pm_rules_show, use_container_width=True)
                else:
                    st.info("No evening rules found above threshold.")


# -------------------------------------------------------------------------------------------------
# PAGE 7: VIVA & EXAM GUIDE
# -------------------------------------------------------------------------------------------------
elif selected_page == "Viva & Exam Guide":
    st.markdown("## 🎓 College Viva & Project Presentation Reference")
    st.markdown("Comprehensive Q&A and theoretical defense guide designed for college data mining examinations.")

    render_disclaimer(is_synthetic=st.session_state.get("is_synthetic", True))

    with st.expander("Q1: Why does Apriori slow down dramatically at lower support thresholds?", expanded=True):
        st.markdown(
            """
            **Answer:**
            Apriori relies on **candidate generation** ($L_{k-1} \times L_{k-1} \rightarrow C_k$).
            At lower support thresholds:
            1. The number of frequent 1-itemsets and 2-itemsets grows substantially.
            2. Deriving candidate 3-itemsets and 4-itemsets causes a **combinatorial explosion**:
               $$\\binom{|L_1|}{k}$$
            3. For every candidate set, Apriori must execute a **full scan of the entire dataset** to count occurrences.
               This incurs heavy I/O overhead and hash-tree matching costs.
            """
        )

    with st.expander("Q2: How does FP-Growth eliminate candidate generation?", expanded=True):
        st.markdown(
            """
            **Answer:**
            FP-Growth uses a **divide-and-conquer** approach:
            1. **Scan 1:** Count item frequencies and sort items in descending order of frequency.
            2. **Scan 2:** Insert all transactions into a highly compressed **FP-Tree** (Frequent Pattern Tree) where overlapping prefixes share common tree nodes.
            3. **Recursive Mining:** For each item starting from the bottom of the header table, it constructs a **Conditional Pattern Base** (sub-paths leading to that item) and builds a small **Conditional FP-Tree**.
            4. Frequent itemsets are directly generated from these recursive prefix trees **without ever testing candidate permutations**.
            """
        )

    with st.expander("Q3: What is the significance of Lift > 1, Lift = 1, and Lift < 1?"):
        st.markdown(
            r"""
            **Answer:**
            - **Lift > 1 (Positive Correlation):** Products $X$ and $Y$ occur together more frequently than expected by random chance. Customers who purchase $X$ are more inclined to also buy $Y$.
            - **Lift = 1 (Statistical Independence):** Product $X$ and $Y$ occurrences are independent ($P(X \cap Y) = P(X)P(Y)$). Bundling them provides no associative synergy.
            - **Lift < 1 (Negative Correlation / Substitutes):** The presence of $X$ decreases the likelihood of $Y$ (e.g. buying two competing foaming cleansers).
            """
        )

    with st.expander("Q4: Why is Jaccard Similarity useful alongside Confidence?"):
        st.markdown(
            """
            **Answer:**
            **Confidence is asymmetric:** $\\text{Confidence}(X \\rightarrow Y) \\neq \\text{Confidence}(Y \\rightarrow X)$.
            If $Y$ is an overwhelmingly frequent staple item (e.g. *Gentle Cleanser* in 90% of baskets), any arbitrary product $X$ will have high confidence for $X \\rightarrow Y$, creating misleading rules.

            **Jaccard Similarity is symmetric:**
            $$\\text{Jaccard}(X, Y) = \\frac{|X \\cap Y|}{|X \\cup Y|}$$
            It normalizes for the baseline popularity of both items, rewarding pairs that co-occur with high mutual exclusivity.
            """
        )

    with st.expander("Q5: How does GlowGuide handle ethical and safety considerations in skincare?"):
        st.markdown(
            """
            **Answer:**
            GlowGuide is explicitly positioned as an **e-commerce behavioral analytics tool**, not a clinical diagnostic system:
            1. Clear visual badges distinguish **synthetic data** from real user surveys.
            2. A prominent **medical safety disclaimer** reminds users that statistical co-occurrence does not imply clinical chemical compatibility (e.g. caution with layering retinol and direct acids).
            """
        )
