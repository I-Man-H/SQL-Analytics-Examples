"""
Statistical analysis utilities.

Reusable functions for the kinds of analyses that appear
repeatedly in data science interviews and take-home tasks:
  - Descriptive statistics with outlier flagging
  - Hypothesis testing (t-test, chi-square, Mann-Whitney)
  - Effect size (Cohen's d, Cramér's V)
  - Correlation analysis with significance
  - Distribution fitting
"""

from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Descriptive statistics
# ---------------------------------------------------------------------------

def describe_numeric(series: pd.Series) -> Dict[str, float]:
    """
    Extended descriptive statistics for a numeric series.

    Adds IQR, outlier count (IQR method), skewness, and kurtosis
    to the standard pandas describe() output.
    """
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    outliers = series[(series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)]

    return {
        "count": int(series.count()),
        "mean": round(series.mean(), 4),
        "median": round(series.median(), 4),
        "std": round(series.std(), 4),
        "min": round(series.min(), 4),
        "q1": round(q1, 4),
        "q3": round(q3, 4),
        "max": round(series.max(), 4),
        "iqr": round(iqr, 4),
        "outlier_count": len(outliers),
        "outlier_pct": round(len(outliers) / len(series) * 100, 2),
        "skewness": round(series.skew(), 4),
        "kurtosis": round(series.kurtosis(), 4),
        "missing": int(series.isna().sum()),
        "missing_pct": round(series.isna().mean() * 100, 2),
    }


# ---------------------------------------------------------------------------
# Hypothesis testing
# ---------------------------------------------------------------------------

def two_sample_ttest(
    group_a: pd.Series,
    group_b: pd.Series,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Welch's two-sample t-test (does not assume equal variances).

    Returns the test statistic, p-value, significance decision,
    and Cohen's d effect size.

    Args:
        group_a: Numeric values for group A.
        group_b: Numeric values for group B.
        alpha:   Significance level (default 0.05).

    Returns:
        Dictionary with t_stat, p_value, significant, cohen_d, interpretation.
    """
    t_stat, p_value = stats.ttest_ind(group_a.dropna(), group_b.dropna(), equal_var=False)
    d = cohens_d(group_a, group_b)
    significant = p_value < alpha

    return {
        "t_stat": round(float(t_stat), 4),
        "p_value": round(float(p_value), 6),
        "alpha": alpha,
        "significant": significant,
        "cohen_d": round(d, 4),
        "effect_size": _interpret_cohens_d(d),
        "mean_a": round(float(group_a.mean()), 4),
        "mean_b": round(float(group_b.mean()), 4),
        "mean_diff": round(float(group_a.mean() - group_b.mean()), 4),
        "interpretation": (
            f"Statistically significant difference (p={p_value:.4f} < {alpha})"
            if significant
            else f"No significant difference (p={p_value:.4f} >= {alpha})"
        ),
    }


def chi_square_test(
    contingency_table: pd.DataFrame,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Chi-square test of independence.

    Args:
        contingency_table: pd.crosstab() output or equivalent.
        alpha:             Significance level.

    Returns:
        Dictionary with chi2, p_value, degrees of freedom, Cramér's V.
    """
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    v = cramers_v(contingency_table)
    significant = p_value < alpha

    return {
        "chi2": round(float(chi2), 4),
        "p_value": round(float(p_value), 6),
        "dof": int(dof),
        "alpha": alpha,
        "significant": significant,
        "cramers_v": round(v, 4),
        "effect_size": _interpret_cramers_v(v),
        "interpretation": (
            f"Significant association (p={p_value:.4f} < {alpha})"
            if significant
            else f"No significant association (p={p_value:.4f} >= {alpha})"
        ),
    }


def mann_whitney_test(
    group_a: pd.Series,
    group_b: pd.Series,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Mann-Whitney U test — non-parametric alternative to t-test.

    Use when data is not normally distributed or sample sizes are small.
    """
    u_stat, p_value = stats.mannwhitneyu(
        group_a.dropna(), group_b.dropna(), alternative="two-sided"
    )
    significant = p_value < alpha

    return {
        "u_stat": round(float(u_stat), 4),
        "p_value": round(float(p_value), 6),
        "alpha": alpha,
        "significant": significant,
        "median_a": round(float(group_a.median()), 4),
        "median_b": round(float(group_b.median()), 4),
        "interpretation": (
            f"Significant difference in distributions (p={p_value:.4f} < {alpha})"
            if significant
            else f"No significant difference (p={p_value:.4f} >= {alpha})"
        ),
    }


# ---------------------------------------------------------------------------
# Effect sizes
# ---------------------------------------------------------------------------

def cohens_d(group_a: pd.Series, group_b: pd.Series) -> float:
    """Cohen's d effect size for two independent groups."""
    a, b = group_a.dropna(), group_b.dropna()
    pooled_std = np.sqrt(
        ((len(a) - 1) * a.std() ** 2 + (len(b) - 1) * b.std() ** 2)
        / (len(a) + len(b) - 2)
    )
    return float((a.mean() - b.mean()) / pooled_std) if pooled_std > 0 else 0.0


def cramers_v(contingency_table: pd.DataFrame) -> float:
    """Cramér's V effect size for chi-square tests."""
    chi2 = stats.chi2_contingency(contingency_table)[0]
    n = contingency_table.values.sum()
    min_dim = min(contingency_table.shape) - 1
    return float(np.sqrt(chi2 / (n * min_dim))) if min_dim > 0 else 0.0


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------

def correlation_matrix(
    df: pd.DataFrame,
    method: str = "pearson",
    min_periods: int = 30,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute correlation matrix with p-values.

    Args:
        df:          DataFrame of numeric columns.
        method:      'pearson', 'spearman', or 'kendall'.
        min_periods: Minimum observations required per pair.

    Returns:
        Tuple of (correlation_matrix, p_value_matrix).
    """
    numeric_df = df.select_dtypes(include=np.number)
    corr = numeric_df.corr(method=method, min_periods=min_periods)

    # Compute p-values
    n = len(numeric_df)
    p_values = pd.DataFrame(
        np.ones_like(corr.values), index=corr.index, columns=corr.columns
    )
    for col_a in corr.columns:
        for col_b in corr.columns:
            if col_a != col_b:
                pair = numeric_df[[col_a, col_b]].dropna()
                if len(pair) >= min_periods:
                    if method == "pearson":
                        _, p = stats.pearsonr(pair[col_a], pair[col_b])
                    elif method == "spearman":
                        _, p = stats.spearmanr(pair[col_a], pair[col_b])
                    else:
                        _, p = stats.kendalltau(pair[col_a], pair[col_b])
                    p_values.loc[col_a, col_b] = p

    return corr, p_values


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _interpret_cohens_d(d: float) -> str:
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


def _interpret_cramers_v(v: float) -> str:
    if v < 0.1:
        return "negligible"
    elif v < 0.3:
        return "small"
    elif v < 0.5:
        return "medium"
    else:
        return "large"
