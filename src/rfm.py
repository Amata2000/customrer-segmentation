# src/rfm.py
import pandas as pd
import numpy as np
from datetime import datetime
from src.utils import plot_distribution, save_plot

def calculate_rfm(df, customer_id='customer_id', 
                  transaction_date='transaction_date', 
                  order_value='order_value'):
    """
    Calculate RFM metrics from transaction data
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Transaction data
    customer_id : str
        Column name for customer identifier
    transaction_date : str
        Column name for transaction date
    order_value : str
        Column name for transaction value
        
    Returns:
    --------
    pandas.DataFrame
        RFM table with one row per customer
    """
    # Use current date as reference (max date in data plus 1 day)
    current_date = df[transaction_date].max() + pd.Timedelta(days=1)
    
    # Calculate RFM metrics
    rfm_table = df.groupby(customer_id).agg({
        transaction_date: lambda x: (current_date - x.max()).days,  # Recency
        customer_id: 'count',  # Frequency
        order_value: 'sum'  # Monetary
    }).rename(columns={
        transaction_date: 'recency',
        customer_id: 'frequency',
        order_value: 'monetary'
    })
    
    # Reset index
    rfm_table = rfm_table.reset_index()
    
    print(f"RFM table created for {rfm_table.shape[0]} customers")
    return rfm_table

def score_rfm(rfm_table, quantiles=[0.2, 0.4, 0.6, 0.8]):
    """
    Score RFM metrics (1-5 scale)
    
    Parameters:
    -----------
    rfm_table : pandas.DataFrame
        RFM metrics table
    quantiles : list
        Quantiles for scoring (default quintiles)
        
    Returns:
    --------
    pandas.DataFrame
        RFM table with scores and combined score
    """
    rfm_scored = rfm_table.copy()
    
    # Recency: lower is better (1 for most recent, 5 for least recent)
    # We'll reverse the score since lower recency is better
    rfm_scored['R_score'] = pd.qcut(rfm_scored['recency'], 
                                    q=5, labels=[5, 4, 3, 2, 1])
    
    # Frequency: higher is better (5 for most frequent, 1 for least frequent)
    rfm_scored['F_score'] = pd.qcut(rfm_scored['frequency'], 
                                    q=5, labels=[1, 2, 3, 4, 5])
    
    # Monetary: higher is better (5 for highest spend, 1 for lowest spend)
    rfm_scored['M_score'] = pd.qcut(rfm_scored['monetary'], 
                                    q=5, labels=[1, 2, 3, 4, 5])
    
    # Convert to numeric
    rfm_scored['R_score'] = rfm_scored['R_score'].astype(int)
    rfm_scored['F_score'] = rfm_scored['F_score'].astype(int)
    rfm_scored['M_score'] = rfm_scored['M_score'].astype(int)
    
    # Create combined RFM score
    rfm_scored['RFM_score'] = rfm_scored['R_score'].astype(str) + \
                              rfm_scored['F_score'].astype(str) + \
                              rfm_scored['M_score'].astype(str)
    
    # Calculate RFM score total
    rfm_scored['RFM_total'] = rfm_scored['R_score'] + \
                              rfm_scored['F_score'] + \
                              rfm_scored['M_score']
    
    return rfm_scored

def plot_rfm_distributions(rfm_table):
    """
    Plot distributions of RFM metrics
    
    Parameters:
    -----------
    rfm_table : pandas.DataFrame
        RFM metrics table
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Recency distribution
    axes[0].hist(rfm_table['recency'], bins=20, edgecolor='black', alpha=0.7, color='skyblue')
    axes[0].set_title('Recency Distribution', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Days Since Last Purchase')
    axes[0].set_ylabel('Frequency')
    axes[0].grid(True, alpha=0.3)
    
    # Frequency distribution
    axes[1].hist(rfm_table['frequency'], bins=20, edgecolor='black', alpha=0.7, color='lightgreen')
    axes[1].set_title('Frequency Distribution', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Number of Transactions')
    axes[1].set_ylabel('Frequency')
    axes[1].grid(True, alpha=0.3)
    
    # Monetary distribution
    axes[2].hist(rfm_table['monetary'], bins=20, edgecolor='black', alpha=0.7, color='salmon')
    axes[2].set_title('Monetary Distribution', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Total Spend ($)')
    axes[2].set_ylabel('Frequency')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_plot(fig, 'rfm_distributions.png')
    return fig

def segment_rfm_by_score(rfm_scored):
    """
    Create business segments based on RFM scores
    
    Parameters:
    -----------
    rfm_scored : pandas.DataFrame
        RFM table with scores
        
    Returns:
    --------
    pandas.DataFrame
        RFM table with segment labels
    """
    rfm_segmented = rfm_scored.copy()
    
    # Define segment mapping based on RFM total score
    conditions = [
        rfm_segmented['RFM_total'] >= 12,  # Champions
        rfm_segmented['RFM_total'] >= 9,   # Loyal Customers
        rfm_segmented['RFM_total'] >= 6,   # Potential Loyalists
        rfm_segmented['RFM_total'] >= 3,   # At Risk
        True                               # Lost Customers
    ]
    
    choices = [
        'Champions',
        'Loyal Customers',
        'Potential Loyalists',
        'At Risk',
        'Lost Customers'
    ]
    
    rfm_segmented['rfm_segment'] = np.select(conditions, choices)
    
    # More detailed segmentation based on individual scores
    def detailed_segment(row):
        if row['R_score'] >= 4 and row['F_score'] >= 4 and row['M_score'] >= 4:
            return 'Platinum Customers'
        elif row['R_score'] >= 3 and row['F_score'] >= 3 and row['M_score'] >= 3:
            return 'Gold Customers'
        elif row['R_score'] >= 2 and row['F_score'] >= 2:
            return 'Silver Customers'
        elif row['R_score'] <= 2 and row['F_score'] <= 2 and row['M_score'] <= 2:
            return 'Inactive Customers'
        elif row['R_score'] <= 2:
            return 'At Risk'
        elif row['F_score'] >= 4:
            return 'Frequent Buyers'
        elif row['M_score'] >= 4:
            return 'Big Spenders'
        else:
            return 'Regular Customers'
    
    rfm_segmented['detailed_segment'] = rfm_segmented.apply(detailed_segment, axis=1)
    
    return rfm_segmented