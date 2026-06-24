-- ============================================================
-- E-commerce analytical queries
-- Demonstrates: aggregations, CTEs, subqueries, JOINs, CASE
-- ============================================================

-- ------------------------------------------------------------
-- 1. Monthly revenue and order volume trend
-- ------------------------------------------------------------
SELECT
    DATE_TRUNC('month', o.order_date)        AS month,
    COUNT(DISTINCT o.order_id)               AS total_orders,
    COUNT(DISTINCT o.customer_id)            AS unique_customers,
    ROUND(SUM(oi.quantity * oi.unit_price), 2) AS gross_revenue,
    ROUND(
        SUM(oi.quantity * oi.unit_price)
        / COUNT(DISTINCT o.order_id), 2
    )                                        AS avg_order_value
FROM orders o
JOIN order_items oi ON o.order_id = oi.order_id
WHERE o.status = 'completed'
GROUP BY 1
ORDER BY 1;


-- ------------------------------------------------------------
-- 2. Revenue and margin by product category
-- ------------------------------------------------------------
WITH category_metrics AS (
    SELECT
        p.category,
        p.subcategory,
        SUM(oi.quantity)                              AS units_sold,
        ROUND(SUM(oi.quantity * oi.unit_price), 2)   AS revenue,
        ROUND(SUM(oi.quantity * p.cost_price), 2)    AS total_cost,
        ROUND(
            (SUM(oi.quantity * oi.unit_price) - SUM(oi.quantity * p.cost_price))
            / NULLIF(SUM(oi.quantity * oi.unit_price), 0) * 100, 2
        )                                             AS margin_pct
    FROM order_items oi
    JOIN products p    ON oi.product_id = p.product_id
    JOIN orders o      ON oi.order_id   = o.order_id
    WHERE o.status = 'completed'
    GROUP BY 1, 2
)
SELECT
    category,
    subcategory,
    units_sold,
    revenue,
    total_cost,
    margin_pct,
    ROUND(revenue / SUM(revenue) OVER (PARTITION BY category) * 100, 2) AS pct_of_category
FROM category_metrics
ORDER BY category, revenue DESC;


-- ------------------------------------------------------------
-- 3. Customer segmentation by lifetime value
-- ------------------------------------------------------------
WITH customer_ltv AS (
    SELECT
        c.customer_id,
        c.name,
        c.country,
        c.segment,
        COUNT(DISTINCT o.order_id)                      AS total_orders,
        ROUND(SUM(oi.quantity * oi.unit_price), 2)      AS lifetime_value,
        MIN(o.order_date)                               AS first_order,
        MAX(o.order_date)                               AS last_order,
        DATE_DIFF('day', MIN(o.order_date), MAX(o.order_date)) AS customer_tenure_days
    FROM customers c
    JOIN orders o      ON c.customer_id = o.customer_id
    JOIN order_items oi ON o.order_id   = oi.order_id
    WHERE o.status = 'completed'
    GROUP BY 1, 2, 3, 4
)
SELECT
    *,
    NTILE(4) OVER (ORDER BY lifetime_value)  AS ltv_quartile,
    CASE
        WHEN lifetime_value >= PERCENTILE_CONT(0.9) OVER () THEN 'Champions'
        WHEN lifetime_value >= PERCENTILE_CONT(0.7) OVER () THEN 'Loyal'
        WHEN lifetime_value >= PERCENTILE_CONT(0.4) OVER () THEN 'Potential'
        ELSE 'At Risk'
    END                                      AS rfm_segment
FROM customer_ltv
ORDER BY lifetime_value DESC;


-- ------------------------------------------------------------
-- 4. Top 10 products by revenue with return rate
-- ------------------------------------------------------------
WITH product_sales AS (
    SELECT
        p.product_id,
        p.name,
        p.category,
        SUM(CASE WHEN o.status = 'completed' THEN oi.quantity ELSE 0 END) AS units_sold,
        SUM(CASE WHEN o.status = 'returned'  THEN oi.quantity ELSE 0 END) AS units_returned,
        ROUND(SUM(CASE WHEN o.status = 'completed'
                       THEN oi.quantity * oi.unit_price ELSE 0 END), 2)   AS revenue
    FROM products p
    JOIN order_items oi ON p.product_id = oi.product_id
    JOIN orders o       ON oi.order_id  = o.order_id
    GROUP BY 1, 2, 3
)
SELECT
    name,
    category,
    units_sold,
    revenue,
    ROUND(units_returned / NULLIF(units_sold, 0) * 100, 2) AS return_rate_pct
FROM product_sales
ORDER BY revenue DESC
LIMIT 10;


-- ------------------------------------------------------------
-- 5. Sales channel performance comparison
-- ------------------------------------------------------------
SELECT
    o.channel,
    COUNT(DISTINCT o.order_id)                      AS orders,
    COUNT(DISTINCT o.customer_id)                   AS customers,
    ROUND(SUM(oi.quantity * oi.unit_price), 2)      AS revenue,
    ROUND(AVG(oi.quantity * oi.unit_price), 2)      AS avg_item_value,
    ROUND(
        SUM(oi.quantity * oi.unit_price)
        / COUNT(DISTINCT o.order_id), 2
    )                                               AS avg_order_value,
    ROUND(
        SUM(oi.quantity * oi.unit_price)
        / SUM(SUM(oi.quantity * oi.unit_price)) OVER () * 100, 2
    )                                               AS revenue_share_pct
FROM orders o
JOIN order_items oi ON o.order_id = oi.order_id
WHERE o.status = 'completed'
GROUP BY 1
ORDER BY revenue DESC;
