-- ============================================================
-- Cohort retention analysis
-- Classic interview question: "How do you measure retention?"
-- ============================================================

-- ------------------------------------------------------------
-- Step 1: Assign each customer to their acquisition cohort
--         (the month of their first completed order)
-- ------------------------------------------------------------
WITH first_orders AS (
    SELECT
        customer_id,
        MIN(order_date) AS first_order_date,
        DATE_TRUNC('month', MIN(order_date)) AS cohort_month
    FROM orders
    WHERE status = 'completed'
    GROUP BY 1
),

-- ------------------------------------------------------------
-- Step 2: For each subsequent order, compute how many months
--         after the cohort month it occurred
-- ------------------------------------------------------------
order_cohorts AS (
    SELECT
        o.customer_id,
        fo.cohort_month,
        DATE_TRUNC('month', o.order_date) AS order_month,
        DATE_DIFF('month', fo.cohort_month, DATE_TRUNC('month', o.order_date)) AS months_since_first
    FROM orders o
    JOIN first_orders fo ON o.customer_id = fo.customer_id
    WHERE o.status = 'completed'
),

-- ------------------------------------------------------------
-- Step 3: Count distinct active customers per cohort per period
-- ------------------------------------------------------------
cohort_counts AS (
    SELECT
        cohort_month,
        months_since_first,
        COUNT(DISTINCT customer_id) AS active_customers
    FROM order_cohorts
    GROUP BY 1, 2
),

-- ------------------------------------------------------------
-- Step 4: Get cohort size (period 0 = acquisition month)
-- ------------------------------------------------------------
cohort_sizes AS (
    SELECT
        cohort_month,
        active_customers AS cohort_size
    FROM cohort_counts
    WHERE months_since_first = 0
)

-- ------------------------------------------------------------
-- Step 5: Calculate retention rate for each period
-- ------------------------------------------------------------
SELECT
    cc.cohort_month,
    cs.cohort_size,
    cc.months_since_first,
    cc.active_customers,
    ROUND(cc.active_customers / cs.cohort_size * 100, 1) AS retention_pct
FROM cohort_counts cc
JOIN cohort_sizes cs ON cc.cohort_month = cs.cohort_month
ORDER BY cc.cohort_month, cc.months_since_first;
