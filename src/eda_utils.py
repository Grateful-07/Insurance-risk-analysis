# src/eda_utils.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def generate_descriptive_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Returns numerical profiling parameters for key financial assets."""
    target_cols = ['TotalPremium', 'TotalClaims', 'Loss_Ratio', 'Margin', 'CustomValueEstimate']
    available_cols = [col for col in target_cols if col in df.columns]
    return df[available_cols].describe().T

def plot_loss_ratio_by_category(df: pd.DataFrame, categorical_col: str, output_path: str):
    """Generates an explicit bar evaluation of portfolio Loss Ratios grouped by customer dimensions."""
    plt.figure(figsize=(10, 5))
    
    # Calculate group aggregated Loss Ratios safely
    group_stats = df.groupby(categorical_col).agg({'TotalPremium': 'sum', 'TotalClaims': 'sum'}).reset_index()
    group_stats['Loss_Ratio'] = group_stats['TotalClaims'] / group_stats['TotalPremium']
    group_stats = group_stats.sort_values(by='Loss_Ratio', ascending=False)
    
    sns.barplot(data=group_stats, x=categorical_col, y='Loss_Ratio', palette='viridis')
    plt.axhline(y=1.0, color='red', linestyle='--', label='Breakeven Point (Loss Ratio = 1.0)')
    plt.title(f'ACIS Portfolio Loss Ratio Distribution across {categorical_col}', fontsize=12, fontweight='bold')
    plt.ylabel('Loss Ratio (Claims / Premiums)')
    plt.xlabel(categorical_col)
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()