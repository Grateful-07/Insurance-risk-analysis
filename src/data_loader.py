# src/data_loader.py
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_insurance_data(file_path: str) -> pd.DataFrame:
    """
    Ingests the ACIS historical claim dataset securely with defensive type-casting.
    Calculates primary derived risk metrics: Loss Ratio and Margin.
    """
    try:
        logging.info(f"Initiating structural data ingestion from: {file_path}")
        df = pd.read_csv(file_path)
        
        if df.empty:
            raise ValueError("The target dataset contains no record parameters.")
            
        # Ensure target financial variables are cast to numeric representations
        df['TotalPremium'] = pd.to_numeric(df['TotalPremium'], errors='coerce').fillna(0)
        df['TotalClaims'] = pd.to_numeric(df['TotalClaims'], errors='coerce').fillna(0)
        
        # Calculate crucial derived insurance baseline metrics
        logging.info("Calculating derived insurance portfolio risk metrics...")
        df['Loss_Ratio'] = df.apply(
            lambda row: row['TotalClaims'] / row['TotalPremium'] if row['TotalPremium'] > 0 else 0, 
            axis=1
        )
        df['Margin'] = df['TotalPremium'] - df['TotalClaims']
        
        logging.info(f"Ingestion successful. Row count: {df.shape[0]}, Feature columns: {df.shape[1]}")
        return df
        
    except FileNotFoundError:
        logging.critical(f"Critical Path Interrupted: File not found at {file_path}")
        raise
    except Exception as e:
        logging.error(f"Unexpected structural data anomaly: {str(e)}")
        raise