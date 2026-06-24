-- ============================================================
-- Window function demonstrations
-- Demonstrates: ROW_NUMBER, RANK, DENSE_RANK, LAG/LEAD,
--               running totals, moving averages, percentiles
-- ============================================================

-- ------------------------------------------------------------
-- 1. Rank customers by revenue within each country
-- ------------------------------------------------------------
WITH customer_revenue AS (
    SELECT
        c.customer_id,
        c.name,
        c.country,
        ROUND(SUM(oi.quantity * oi.unit_price), 2) AS total_revenue
    FROM customers c
    JOIN orders o       ON c.customer_id = o.customer_id
    JOIN order_items oi ON o.order_id    = oi.order_id
    WHERE o.status = 'completed'
    GROUP BY 1, 2, 3
)
SELECT
    country,
    name,
    total_revenue,
    RANK()       OVER (PARTITION BY country ORDER BY total_revenue DESC) AS revenue_rank,
    DENSE_RANK() OVER (PARTITION BY country ORDER BY total_revenue DESC) AS revenue_dense_rank,
    ROUND(
        total_revenue / SUM(total_revenue) OVER (PARTITION BY country) * 100, 2
    )                                                                    AS pct_of_country_revenue
FROM customer_revenue
ORDER BY country, revenue_rank;


-- ------------------------------------------------------------
-- 2. Month-over-month revenue growth with LAG
-- ------------------------------------------------------------
WITH monthly_revenue AS (
    SELECT
        DATE_TRUNC('month', o.order_date)           AS month,
        ROUND(SUM(oi.quantity * oi.unit_price), 2)  AS revenue
    FROM orders o
    JOIN order_items oi ON o.order_id = oi.order_id
    WHERE o.status = 'completed'
    GROUP BY 1
)
SELECT
    month,
    revenue,
    LAG(revenue) OVER (ORDER BY month)           AS prev_month_revenue,
    ROUND(
        (revenue - LAG(revenue) OVER (ORDER BY month))
        / NULLIF(LAG(revenue) OVER (ORDER BY month), 0) * 100, 2
    )                                            AS mom_growth_pct,
    SUM(revenue) OVER (ORDER BY month
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cumulative_revenue,
    AVG(revenue) OVER (ORDER BY month
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS rolling_3m_avg
FROM monthly_revenue
ORDER BY month;


-- ------------------------------------------------------------
-- 3. Running total and percentile of order value
-- ------------------------------------------------------------
SELECT
    order_id,
    order_date,
    order_value,
    SUM(order_value) OVER (ORDER BY order_date
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)  AS running_total,
    ROUND(PERCENT_RANK() OVER (ORDER BY order_value) * 100, 2) AS percentile,
    NTILE(10) OVER (ORDER BY order_value)                  AS decile
FROM (
    SELECT
        o.order_id,
        o.order_date,
        ROUND(SUM(oi.quantity * oi.unit_price), 2) AS order_value
    FROM orders o
    JOIN order_items oi ON o.order_id = oi.order_id
    WHERE o.status = 'completed'
    GROUP BY 1, 2
) sub
ORDER BY order_date;


-- ------------------------------------------------------------
-- 4. First and most recent purchase per customer (FIRST_VALUE / LAST_VALUE)
-- ------------------------------------------------------------
SELECT DISTINCT
    customer_id,
    FIRST_VALUE(order_date) OVER (
        PARTITION BY customer_id ORDER BY order_date
        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    ) AS first_purchase_date,
    LAST_VALUE(order_date) OVER (
        PARTITION BY customer_id ORDER BY order_date
        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    ) AS last_purchase_date,
    COUNT(*) OVER (PARTITION BY customer_id) AS total_orders,
    DATE_DIFF('day',
        FIRST_VALUE(order_date) OVER (
            PARTITION BY customer_id ORDER BY order_date
            ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING),
        LAST_VALUE(order_date) OVER (
            PARTITION BY customer_id ORDER BY order_date
            ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING)
    ) AS days_between_first_last
FROM orders
WHERE status = 'completed'
ORDER BY total_orders DESC;
