"""Tests for statistical utility functions."""

import numpy as np
import pandas as pd
import pytest
from utils.stats import (
    describe_numeric, two_sample_ttest,
    mann_whitney_test, chi_square_test,
    cohens_d, cramers_v,
)

np.random.seed(0)
GROUP_A = pd.Series(np.random.normal(50, 10, 100))
GROUP_B = pd.Series(np.random.normal(60, 10, 100))  # clearly different
GROUP_C = pd.Series(np.random.normal(50.1, 10, 100))  # nearly identical


class TestDescribeNumeric:
    def test_returns_all_keys(self):
        result = describe_numeric(GROUP_A)
        expected = ["count","mean","median","std","min","q1","q3","max",
                    "iqr","outlier_count","outlier_pct","skewness","kurtosis",
                    "missing","missing_pct"]
        for key in expected:
            assert key in result

    def test_count_matches(self):
        assert describe_numeric(GROUP_A)["count"] == 100

    def test_handles_missing_values(self):
        s = GROUP_A.copy()
        s.iloc[:10] = np.nan
        result = describe_numeric(s)
        assert result["missing"] == 10
        assert result["missing_pct"] == 10.0


class TestTTest:
    def test_detects_significant_difference(self):
        result = two_sample_ttest(GROUP_A, GROUP_B)
        assert result["significant"] is True
        assert result["p_value"] < 0.05

    def test_no_significant_difference(self):
        result = two_sample_ttest(GROUP_A, GROUP_C)
        assert result["significant"] is False

    def test_result_keys(self):
        result = two_sample_ttest(GROUP_A, GROUP_B)
        for key in ["t_stat","p_value","significant","cohen_d","effect_size","interpretation"]:
            assert key in result

    def test_effect_size_label(self):
        result = two_sample_ttest(GROUP_A, GROUP_B)
        assert result["effect_size"] in ["negligible","small","medium","large"]


class TestMannWhitney:
    def test_detects_significant_difference(self):
        result = mann_whitney_test(GROUP_A, GROUP_B)
        assert result["significant"] is True

    def test_result_has_medians(self):
        result = mann_whitney_test(GROUP_A, GROUP_B)
        assert "median_a" in result
        assert "median_b" in result


class TestEffectSizes:
    def test_cohens_d_large(self):
        a = pd.Series(np.random.normal(0, 1, 200))
        b = pd.Series(np.random.normal(3, 1, 200))
        assert cohens_d(a, b) > 0.8

    def test_cohens_d_zero_variance(self):
        a = pd.Series([5.0] * 50)
        b = pd.Series([5.0] * 50)
        assert cohens_d(a, b) == 0.0

    def test_cramers_v_range(self):
        ct = pd.crosstab(
            pd.Series(np.random.choice(["A","B","C"], 200)),
            pd.Series(np.random.choice(["X","Y"], 200)),
        )
        v = cramers_v(ct)
        assert 0.0 <= v <= 1.0
