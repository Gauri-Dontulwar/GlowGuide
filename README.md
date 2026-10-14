# 🌿 GlowGuide: Comparative Analysis of Apriori and FP-Growth for Skincare Product Association Mining

[![Python Version](https://img.shields.io/badge/Python-3.13-4A6B5B.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Streamlit-1.65.0-688C7A.svg)](https://streamlit.io/)
[![Algorithms](https://img.shields.io/badge/Algorithms-Apriori%20%7C%20FP--Growth-DDA7A5.svg)](http://rasbt.github.io/mlxtend/)
[![License](https://img.shields.io/badge/License-Academic%20Demo-2E4F3E.svg)]()

> An end-to-end data mining and market basket analytics system built for college-level demonstrations, academic research, and e-commerce product affinity modeling in the cosmetic and skincare industry.

---

## 📖 Executive Summary & Motivation

In modern retail and personal wellness, skincare purchases are rarely isolated events. Consumers curate multi-step regimens—pairing cleansers, active serums, moisturizers, and sunscreens tailored to specific times of day (**Morning / AM** vs. **Evening / PM**) and dermatological profiles (**Dry, Oily, Sensitive, Combination**).

**GlowGuide** demonstrates how frequent itemset mining and association rule discovery extract high-value consumer co-occurrences from transactional baskets. It features an interactive, skincare-inspired Streamlit dashboard styled in **warm cream (`#FAF8F5`)**, **botanical sage green (`#4A6B5B`)**, and **soft rose-blush (`#DDA7A5`)**, enabling real-time comparative benchmarks between the classical **Apriori** algorithm and the high-performance **FP-Growth** (Frequent Pattern Growth) algorithm.

---

## ⚖️ Theoretical Foundation: Apriori vs. FP-Growth

| Dimension | Apriori Algorithm (Agrawal & Srikant, 1994) | FP-Growth Algorithm (Han, Pei & Yin, 2000) |
| :--- | :--- | :--- |
| **Search Paradigm** | Breadth-First Search (Level-wise generation) | Depth-First Search (Divide and conquer) |
| **Candidate Generation** | **Yes** — Combinatorial candidate sets ($L_{k-1} \to C_k$) | **No** — Eliminates candidate generation entirely |
| **Database Scans** | Multiple scans: $k_{max} + 1$ full passes | **Exactly 2 passes** across the dataset |
| **Data Structure** | Array / Hash-Tree of candidate itemsets | Compact in-memory **FP-Tree** with linked header table |
| **Time Complexity** | $O(2^{\|Items\|})$ in dense or low-support scenarios | $O(\|Transactions\| \times \|Items\|)$ |
| **Space Complexity** | High memory demand during candidate explosions | Moderate to low (stores frequent item prefix paths) |
| **Bottleneck** | Disk I/O scans & candidate generation overhead | Memory overhead if transactions are extremely sparse |

### Why FP-Growth Outperforms Apriori
1. **Zero Candidate Generation:** Apriori generates millions of candidate pairs at low support levels. FP-Growth extracts frequent patterns directly from conditional FP-trees.
2. **Reduced I/O:** Pass 1 gathers item support counts. Pass 2 builds the prefix tree. All subsequent mining occurs in memory.

---

## 📐 Mathematical Formulation of Association Metrics

### 1. Support
The proportion of transactions in dataset $D$ containing both itemsets $X$ and $Y$:
$$\text{Support}(X \Rightarrow Y) = P(X \cap Y) = \frac{\sigma(X \cup Y)}{|D|}$$

### 2. Confidence
The conditional probability that a basket contains $Y$ given that it already contains $X$:
$$\text{Confidence}(X \Rightarrow Y) = P(Y \mid X) = \frac{\text{Support}(X \cup Y)}{\text{Support}(X)}$$

### 3. Lift
Measures how much more often $X$ and $Y$ occur together than expected under statistical independence:
$$\text{Lift}(X \Rightarrow Y) = \frac{P(X \cap Y)}{P(X) \cdot P(Y)} = \frac{\text{Confidence}(X \Rightarrow Y)}{\text{Support}(Y)}$$
- $\text{Lift} > 1$: Positive association (items complement each other).
- $\text{Lift} = 1$: Statistical independence (no bundling synergy).
- $\text{Lift} < 1$: Negative association / substitute products.

### 4. Leverage
Quantifies the surplus co-occurrence above expected independent co-occurrence:
$$\text{Leverage}(X \Rightarrow Y) = P(X \cap Y) - P(X)P(Y)$$

### 5. Conviction
Measures the dependency of the rule error rate against baseline independence:
$$\text{Conviction}(X \Rightarrow Y) = \frac{1 - P(Y)}{1 - \text{Confidence}(X \Rightarrow Y)}$$

### 6. Jaccard Similarity (Symmetric Metric)
Normalizes for the baseline popularity of both items to prevent misleading rules caused by universally popular staples:
$$\text{Jaccard}(X, Y) = \frac{|X \cap Y|}{|X \cup Y|} = \frac{\text{Support}(X \cup Y)}{\text{Support}(X) + \text{Support}(Y) - \text{Support}(X \cup Y)}$$

---

## ☀️🌙 Routine Segmentation (Chrono-Skincare Analysis)

GlowGuide partitions transactions by routine context to evaluate how basket affinity shifts between daytime environmental defense and nighttime cellular recovery:

- **Morning (AM) Regimens:**
  - *Core Pattern:* Gentle Cleanser $\to$ 15% Vitamin C Antioxidant Serum $\to$ Hyaluronic Acid $\to$ Broad Spectrum SPF 50.
  - *Dermatological Rationale:* Photoprotection against UV-induced reactive oxygen species (ROS) and environmental oxidative stress.
- **Evening (PM) Regimens:**
  - *Core Pattern:* Oil Cleansing Balm $\to$ Hydrating Cream Cleanser (Double Cleanse) $\to$ Retinol 0.5% / AHA-BHA $\to$ Ceramide Barrier Repair Cream.
  - *Dermatological Rationale:* Accelerated nocturnal cellular turnover, dermal repair, and barrier replenishment.

---

## 🗂️ Project Architecture & File Structure

```
d:/GlowGuide/
├── .streamlit/
│   └── config.toml           # Streamlit cream & sage-green theme configuration
├── data/
│   └── synthetic_skincare_transactions.csv  # Pre-generated 1,250-transaction demo dataset
├── src/
│   ├── __init__.py           # Package marker
│   ├── theme.py              # Custom CSS tokens, hero banner, KPI cards, and badges
│   ├── data_loader.py        # Synthetic dataset generator, CSV format auto-detection & cleaning
│   ├── mining.py             # Apriori & FP-Growth engines, rule generation, Jaccard, benchmarking
│   ├── visualizations.py     # Plotly interactive charts (runtimes, scatter landscape, heatmap)
│   └── routine_analysis.py   # AM vs PM segmentation and chrono-skincare rule extraction
├── tests/
│   └── test_pipeline.py      # Automated unit & integration tests (100% pass rate)
├── app.py                    # Main Streamlit dashboard application
├── requirements.txt          # Python dependencies with pinned prebuilt wheels
└── README.md                 # Detailed college project documentation & guide
```

---

## 💻 System Compatibility & Windows ARM64 Considerations

This project targets **Windows on ARM64** with **Python 3.13**:
- On Windows ARM64, native compilation of libraries like `pyarrow` or `scipy` can trigger errors requiring `Microsoft Visual C++ Build Tools`.
- GlowGuide solves this by utilizing **Python 3.13 (AMD64)** via Windows 11's built-in x64 translation engine (Microsoft Prism).
- All dependencies (`numpy`, `pandas`, `streamlit`, `mlxtend`, `scipy`, `matplotlib`, `plotly`) install from **100% prebuilt binary wheels** with zero C++ compilation required.

---

## 🚀 Step-by-Step Installation & Running Guide

### 1. Open Terminal in VS Code
Open the project directory in VS Code:
```powershell
cd D:\GlowGuide
```

### 2. Create and Activate Virtual Environment
Use the standard 64-bit Python 3.13 installation:
```powershell
# Create virtual environment
& "C:\Users\gauri\AppData\Local\Programs\Python\Python313\python.exe" -m venv .venv

# Activate in PowerShell
.venv\Scripts\Activate.ps1
```
*(If PowerShell restricts script execution, run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned`)*

### 3. Install Dependencies
Install all required libraries from `requirements.txt`:
```powershell
pip install -r requirements.txt
```

### 4. Run Automated Test Suite
Verify that all 8 unit and pipeline tests pass:
```powershell
python -m unittest tests/test_pipeline.py
```
*Expected output: `Ran 8 tests ... OK`*

### 5. Launch the Streamlit Dashboard
Start the web application:
```powershell
streamlit run app.py
```
The dashboard will open automatically in your browser at:
👉 **`http://localhost:8501`**

---

## 🎓 College Viva & Presentation Guide

When demonstrating this project to professors, evaluators, or classmates:

1. **Start with the Home & Overview Page:**
   - Introduce the business and retail context: multi-step skincare routines naturally form high-affinity market baskets.
   - Show the prominent **Academic & Medical Safety Disclaimer**.
2. **Move to the Dataset & Ingestion Page:**
   - Explain that GlowGuide works out of the box using **1,250 synthetic transactions** modeling realistic archetypes, but also accepts user-uploaded CSV files.
   - Point out the data cleaning steps (deduplication, casing normalization, minimum basket size filter).
3. **Showcase the Algorithm Comparison Page:**
   - Run the **Live Comparative Test**: Show that FP-Growth executes significantly faster than Apriori while finding the exact same number of frequent itemsets (**100% mathematical equivalence**).
   - Click **Run Multi-Threshold Benchmark Sweep**: Highlight how Apriori's runtime curve curves upward exponentially at low support thresholds due to candidate generation overhead, while FP-Growth stays nearly linear.
4. **Demonstrate Association Rules:**
   - Inspect the **Support vs. Confidence Scatter Plot** colored by Lift.
   - Open the **Plain-English Rule Translator** and explain how a cosmetic brand manager uses the discovered Lift and Jaccard metrics to design product bundles.
5. **Demonstrate AM vs. PM Routine Segmentation:**
   - Highlight the contrast between daytime photoprotection (Vitamin C + SPF 50) and nighttime cell renewal (Retinol + Ceramides).

---

## 🌿 Ethical & Dermatological Safety Disclaimer

> [!IMPORTANT]
> **Academic & Retail Demonstration Only:**
> GlowGuide is strictly an educational data-mining application that models consumer co-purchasing habits and self-reported routine surveys.
> - **No Medical Claims:** Co-occurrence in retail transactions does **not** signify clinical safety or biochemical compatibility.
> - **Actives Caution:** Active dermatological ingredients (such as high-strength retinoids and exfoliating direct acids) require careful scheduling to prevent epidermal barrier damage and should only be combined under the guidance of a licensed dermatologist.
> - **Synthetic Attribution:** All demo transactions are clearly labeled as synthetic modeling data.
