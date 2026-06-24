# sql-analytics-portfolio

> **SQL + Python data analysis portfolio — e-commerce, cohort retention, and health outcomes**

[![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)](https://www.python.org/)
[![DuckDB](https://img.shields.io/badge/DuckDB-1.0-yellow?style=flat-square)](https://duckdb.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.2-150458?style=flat-square&logo=pandas)](https://pandas.pydata.org/)
[![SciPy](https://img.shields.io/badge/SciPy-1.14-8CAAE6?style=flat-square)](https://scipy.org/)
[![Tests](https://img.shields.io/badge/Tests-pytest-green?style=flat-square&logo=pytest)](https://pytest.org/)

---

## Overview

Three end-to-end analytical notebooks, each structured around a real business question.
Every notebook follows the same workflow: define the question → write SQL to extract the
data → validate and explore in pandas → test a hypothesis → visualise results → state findings.

All datasets are synthetically generated so every notebook runs out of the box with no
external data dependencies.

---

## Notebooks

### 01 — E-commerce Sales Analysis
**Questions:** What is our monthly revenue trend? Which categories drive margin? How does
average order value differ by channel? Who are our highest-value customers?

**SQL concepts:** Multi-table JOINs, CTEs, GROUP BY aggregations, CASE expressions,
window functions (PERCENTILE_CONT, NTILE), revenue share calculations.

**Python concepts:** DuckDB integration, pandas data manipulation, distribution analysis,
LTV segmentation, matplotlib/seaborn visualisation.

---

### 02 — Cohort Retention Analysis
**Questions:** What percentage of customers return in months 1, 3, 6, 12?
Are newer cohorts retaining better? Where is the natural retention floor?

**SQL concepts:** Multi-step CTE chain, DATE_TRUNC, DATE_DIFF, cohort construction,
self-join pattern, percentage calculations.

**Python concepts:** Pivot table construction, cohort heatmap, retention curve, floor detection.



---

### 03 — Health Outcomes Analysis
**Questions:** Do patients who complete a rehabilitation programme recover faster?
Is age a significant predictor of outcome? Are group differences statistically significant?

**SQL concepts:** Conditional aggregations, GROUP BY on clinical categories, STDDEV.

**Python concepts:** Welch's t-test, Mann-Whitney U test, Cohen's d effect size,
Pearson/Spearman correlation with p-values, non-parametric testing rationale.

---

## Project Structure

```
sql-analytics-portfolio/
├── notebooks/
│   ├── 01_ecommerce_sales_analysis.py       # Revenue, margin, LTV, channel
│   ├── 02_cohort_retention_analysis.py      # Cohort retention heatmap
│   └── 03_health_outcomes_analysis.py       # Clinical hypothesis testing
├── sql/
│   ├── 01_ecommerce_schema.sql              # Table definitions
│   ├── 02_ecommerce_queries.sql             # Annotated analytical queries
│   ├── 03_health_queries.sql                # Clinical data queries
│   ├── 04_window_functions.sql              # Window function reference
│   └── 05_cohort_analysis.sql              # Standalone cohort SQL
├── utils/
│   ├── db.py                                # DuckDB connection & query helpers
│   ├── stats.py                             # Hypothesis tests, effect sizes, correlation
│   └── plotting.py                          # Reusable matplotlib/seaborn functions
├── tests/
│   ├── test_stats.py                        # Unit tests for statistical functions
│   └── test_db.py                           # Unit tests for database utilities
├── data/
│   └── raw/                                 # Place external CSVs here if needed
└── requirements.txt
```

---

## SQL Skills Demonstrated

| Concept | Where |
|---|---|
| Multi-table JOINs | `02_ecommerce_queries.sql`, all notebooks |
| CTEs (multi-step) | `05_cohort_analysis.sql`, notebook 02 |
| Window functions | `04_window_functions.sql`, notebook 01 |
| LAG / LEAD | `04_window_functions.sql` |
| RANK / DENSE_RANK / NTILE | `04_window_functions.sql`, notebook 01 |
| Running totals & moving averages | `04_window_functions.sql` |
| Cohort construction | `05_cohort_analysis.sql`, notebook 02 |
| Conditional aggregation (CASE WHEN) | `02_ecommerce_queries.sql` |
| Percentile functions | `02_ecommerce_queries.sql` |
| DATE functions (TRUNC, DIFF) | All SQL files |

---

## Statistical Methods

| Method | Use case | File |
|---|---|---|
| Descriptive statistics | Distribution summary with IQR outlier detection | `utils/stats.py` |
| Welch's t-test | Group mean comparison (unequal variance) | `utils/stats.py`, notebook 03 |
| Mann-Whitney U | Non-parametric group comparison | `utils/stats.py`, notebook 03 |
| Chi-square test | Categorical association | `utils/stats.py` |
| Cohen's d | Effect size for t-test | `utils/stats.py`, notebook 03 |
| Cramér's V | Effect size for chi-square | `utils/stats.py` |
| Pearson / Spearman correlation | Variable relationships with p-values | `utils/stats.py`, notebook 03 |

---

## Quickstart

```bash
git clone https://github.com/I-Man-H/sql-analytics-portfolio.git
cd sql-analytics-portfolio

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Run notebooks

Convert `.py` files to Jupyter notebooks and launch:

```bash
pip install jupytext
jupytext --to notebook notebooks/01_ecommerce_sales_analysis.py
jupyter notebook
```

Or run directly as scripts:

```bash
python notebooks/01_ecommerce_sales_analysis.py
python notebooks/02_cohort_retention_analysis.py
python notebooks/03_health_outcomes_analysis.py
```

### Run SQL files against DuckDB

```python
import duckdb
con = duckdb.connect()
sql = open("sql/04_window_functions.sql").read()
# Run individual statements as needed
```

### Run tests

```bash
pytest tests/ -v
```

---

## Why DuckDB?

DuckDB is an in-process analytical database, it requires no server, no configuration,
and no Docker. It runs embedded in Python like SQLite but supports the full analytical
SQL feature set: window functions, CTEs, PIVOT, UNNEST, ASOF joins, and direct
reading of CSV and Parquet files. It is increasingly used in data science workflows
as a fast alternative to spinning up a Postgres or Redshift instance for local analysis.

---

## Contact

**Iman Hosseini** — Data Scientist | ML Engineer
[LinkedIn](https://www.linkedin.com/in/i-man-hosseini/) · [Google Scholar](https://scholar.google.com/citations?user=ZBlw7J0AAAAJ) · [GitHub](https://github.com/I-Man-H)
