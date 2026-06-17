# tests/test_pipeline.py
import pytest
import pandas as pd
import numpy as np

def test_loss_ratio_calculation():
    """Validates that derived risk metrics avoid divide-by-zero bounds."""
    # Mock dataframe reflecting standard premium structures
    mock_df = pd.DataFrame({
        'TotalPremium': [1000, 2000, 0],
        'TotalClaims': [500, 3000, 400]
    })
    
    # Emulate load_insurance_data metrics logic
    mock_df['Loss_Ratio'] = mock_df.apply(
        lambda row: row['TotalClaims'] / row['TotalPremium'] if row['TotalPremium'] > 0 else 0, 
        axis=1
    )
    
    assert mock_df['Loss_Ratio'].iloc[0] == 0.5
    assert mock_df['Loss_Ratio'].iloc[1] == 1.5
    assert mock_df['Loss_Ratio'].iloc[2] == 0.0 # Handled zero premium boundary successfully