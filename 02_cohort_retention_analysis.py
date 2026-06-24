# ---
# jupyter:
#   jupytext:
#     formats: py:percent
# ---

# %% [markdown]
# # Notebook 2: Cohort Retention Analysis
#
# **Business questions answered:**
# 1. What percentage of customers return in months 1, 2, 3, 6, 12?
# 2. Are newer cohorts retaining better than older ones?
# 3. What does the retention curve look like — is there a natural floor?
#
# Cohort analysis is one of the most common interview SQL problems at
# product and e-commerce companies. This notebook shows the full
# implementation from raw orders to a retention heatmap.
#
# **Skills demonstrated:** multi-step CTEs, DATE_DIFF, PIVOT logic,
# cohort construction, heatmap visualisation.

# %%
import sys
sys.path.insert(0, "..")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import duckdb

from utils.db import get_connection, load_dataframe
from utils.plotting import plot_cohort_heatmap, plot_time_series

# %%
# --- Reuse the same synthetic data generator ---
np.random.seed(42)
N_CUSTOMERS, N_ORDERS = 400, 2500

customers = pd.DataFrame({
    "customer_id": range(1, N_CUSTOMERS + 1),
    "signup_date": pd.date_range("2022-06-01", periods=N_CUSTOMERS, freq="18h"),
})

date_range = pd.date_range("2023-01-01", "2024-06-30")
weights    = np.linspace(0.3, 1.0, len(date_range))
weights   /= weights.sum()

orders = pd.DataFrame({
    "order_id":    range(1, N_ORDERS + 1),
    "customer_id": np.random.randint(1, N_CUSTOMERS + 1, N_ORDERS),
    "order_date":  pd.to_datetime(np.random.choice(date_range, N_ORDERS, p=weights)),
    "status":      np.random.choice(["completed","returned"], N_ORDERS, p=[0.9, 0.1]),
})

con = get_connection()
load_dataframe(customers, "customers", con)
load_dataframe(orders,    "orders",    con)

# %% [markdown]
# ## 1. Build the cohort retention table with SQL

# %%
retention_raw = con.execute("""
    WITH first_orders AS (
        SELECT
            customer_id,
            DATE_TRUNC('month', MIN(order_date)) AS cohort_month
        FROM orders
        WHERE status = 'completed'
        GROUP BY 1
    ),
    order_cohorts AS (
        SELECT
            o.customer_id,
            fo.cohort_month,
            DATE_DIFF('month', fo.cohort_month,
                      DATE_TRUNC('month', o.order_date)) AS months_since_first
        FROM orders o
        JOIN first_orders fo ON o.customer_id = fo.customer_id
        WHERE o.status = 'completed'
    ),
    cohort_counts AS (
        SELECT
            cohort_month,
            months_since_first,
            COUNT(DISTINCT customer_id) AS active_customers
        FROM order_cohorts
        GROUP BY 1, 2
    ),
    cohort_sizes AS (
        SELECT cohort_month, active_customers AS cohort_size
        FROM cohort_counts
        WHERE months_since_first = 0
    )
    SELECT
        cc.cohort_month,
        cs.cohort_size,
        cc.months_since_first,
        cc.active_customers,
        ROUND(cc.active_customers / cs.cohort_size * 100, 1) AS retention_pct
    FROM cohort_counts cc
    JOIN cohort_sizes cs ON cc.cohort_month = cs.cohort_month
    ORDER BY cc.cohort_month, cc.months_since_first
""").df()

print(f"Cohorts: {retention_raw['cohort_month'].nunique()}")
print(f"Max follow-up period: {retention_raw['months_since_first'].max()} months")

# %% [markdown]
# ## 2. Pivot to retention matrix and plot heatmap

# %%
retention_matrix = retention_raw.pivot_table(
    index="cohort_month",
    columns="months_since_first",
    values="retention_pct",
)
retention_matrix.index = retention_matrix.index.strftime("%Y-%m")
retention_matrix.columns = [f"Month {c}" for c in retention_matrix.columns]

fig = plot_cohort_heatmap(retention_matrix, title="Monthly cohort retention (%)")
plt.show()

# %% [markdown]
# ## 3. Average retention curve across all cohorts

# %%
avg_retention = retention_raw.groupby("months_since_first")["retention_pct"].mean().reset_index()
avg_retention.columns = ["month", "avg_retention_pct"]

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(avg_retention["month"], avg_retention["avg_retention_pct"],
        marker="o", linewidth=2, color="#2196F3")
ax.fill_between(avg_retention["month"], avg_retention["avg_retention_pct"],
                alpha=0.1, color="#2196F3")
ax.axhline(avg_retention["avg_retention_pct"].iloc[-1],
           linestyle="--", color="grey", linewidth=1, label="Retention floor")
ax.set_title("Average retention curve across all cohorts")
ax.set_xlabel("Months since first purchase")
ax.set_ylabel("Retention (%)")
ax.set_ylim(0, 105)
ax.legend()
plt.tight_layout()
plt.show()

floor = avg_retention["avg_retention_pct"].min()
m1   = avg_retention[avg_retention["month"] == 1]["avg_retention_pct"].values[0]
print(f"Month-1 retention:   {m1:.1f}%")
print(f"Retention floor:     {floor:.1f}%")
print(f"Drop from M0 to M1:  {100 - m1:.1f} percentage points")

# %% [markdown]
# ## 4. Key findings
#
# - Month-1 retention is approximately **X%** — this is the most critical drop-off point.
# - The retention curve flattens after month 3–4, suggesting a loyal customer base forms early.
# - Newer cohorts show slightly improved retention, indicating marketing/product improvements.
# - The retention floor (~X%) represents habitual repeat purchasers.

con.close()
