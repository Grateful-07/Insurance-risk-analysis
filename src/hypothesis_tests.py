# src/hypothesis_tests.py
import numpy as np
import pandas as pd
from scipy import stats

def test_claim_frequency(df, group_col, group_a, group_b):
    """
    Tracks CATEGORICAL metrics (Claim Frequency: Proportion of policies with claims).
    Applies a Chi-Squared Test of Independence.
    """
    # Filter dataset for the two target comparison groups
    sub_df = df[df[group_col].isin([group_a, group_b])].copy()
    
    # Create the contingency table: Group vs Claimed
    contingency_table = pd.crosstab(sub_df[group_col], sub_df['Claimed'])
    
    # Run Chi-Square test
    chi2, p_val, dof, expected = stats.chi2_contingency(contingency_table)
    
    # Dynamically find the column label used for positive claims (1, True, or '1')
    claim_col = None
    for possible_label in [1, True, 1.0, '1']:
        if possible_label in contingency_table.columns:
            claim_col = possible_label
            break
            
    # Calculate tracking frequencies safely
    if claim_col is not None:
        freq_a = contingency_table.loc[group_a].get(claim_col, 0) / contingency_table.loc[group_a].sum()
        freq_b = contingency_table.loc[group_b].get(claim_col, 0) / contingency_table.loc[group_b].sum()
    else:
        freq_a = 0.0
        freq_b = 0.0
    
    return {
        "Test Type": "Chi-Squared Test of Independence",
        "p-value": p_val,
        "Group A Freq": freq_a,
        "Group B Freq": freq_b,
        "Decision": "Reject H0" if p_val < 0.05 else "Fail to Reject H0"
    }

def test_numerical_kpi(df, group_col, group_a, group_b, kpi_col, subset_claims_only=False):
    """
    Tracks NUMERICAL metrics (Claim Severity or Net Margin Contribution).
    Applies an Independent Welch's t-test.
    """
    sub_df = df[df[group_col].isin([group_a, group_b])].copy()
    
    # For Severity, filter to records where a claim actually occurred
    if subset_claims_only:
        sub_df = sub_df[sub_df['Claimed'].isin([1, True, 1.0, '1'])]
        
    data_a = sub_df[sub_df[group_col] == group_a][kpi_col].dropna()
    data_b = sub_df[sub_df[group_col] == group_b][kpi_col].dropna()
    
    # Run Welch's t-test
    t_stat, p_val = stats.ttest_ind(data_a, data_b, equal_var=False)
    
    return {
        "Test Type": "Independent Welch's t-test",
        "p-value": p_val,
        "Group A Mean": data_a.mean() if not data_a.empty else 0,
        "Group B Mean": data_b.mean() if not data_b.empty else 0,
        "Decision": "Reject H0" if p_val < 0.05 else "Fail to Reject H0"
    }