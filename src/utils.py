# src/utils.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_data(filepath):
    """
    Load raw transaction data from CSV file
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file
        
    Returns:
    --------
    pandas.DataFrame
        Loaded transaction data
    """
    df = pd.read_csv(filepath)
    print(f"Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def clean_data(df):
    """
    Clean the transaction data
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Raw transaction data
        
    Returns:
    --------
    pandas.DataFrame
        Cleaned transaction data
    """
    df_clean = df.copy()
    
    # Convert date column
    df_clean['transaction_date'] = pd.to_datetime(df_clean['transaction_date'])
    
    # Check for missing values
    missing = df_clean.isnull().sum()
    if missing.sum() > 0:
        print(f"Missing values found:\n{missing[missing > 0]}")
        # Fill or drop missing values as appropriate
        df_clean = df_clean.dropna()
    
    # Check for duplicates
    duplicates = df_clean.duplicated().sum()
    if duplicates > 0:
        print(f"Found {duplicates} duplicate rows, removing them...")
        df_clean = df_clean.drop_duplicates()
    
    # Validate numeric columns
    numeric_cols = ['order_value', 'quantity']
    for col in numeric_cols:
        if col in df_clean.columns:
            # Convert to numeric, coercing errors
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # Remove negative or zero values for order_value
    df_clean = df_clean[df_clean['order_value'] > 0]
    
    print(f"Data cleaned: {df_clean.shape[0]} rows remaining")
    return df_clean

def plot_distribution(data, column, title, bins=30, figsize=(10, 6)):
    """
    Plot distribution of a column
    
    Parameters:
    -----------
    data : pandas.DataFrame or pandas.Series
        Data to plot
    column : str
        Column name
    title : str
        Plot title
    bins : int
        Number of bins for histogram
    figsize : tuple
        Figure size
    """
    plt.figure(figsize=figsize)
    
    if isinstance(data, pd.DataFrame):
        values = data[column]
    else:
        values = data
        
    plt.hist(values, bins=bins, edgecolor='black', alpha=0.7)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel(column, fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt.gcf()

def save_plot(fig, filename):
    """
    Save plot to file
    
    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        Figure object
    filename : str
        Output filename
    """
    fig.savefig(f'visualizations/{filename}', dpi=300, bbox_inches='tight')
    plt.close(fig)