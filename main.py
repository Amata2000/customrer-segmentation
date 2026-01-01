# main.py
"""
Customer Segmentation Project - Main Execution Script
Author: Data Analytics Team
Date: 2024

This script orchestrates the entire customer segmentation pipeline:
1. Data loading and cleaning
2. RFM analysis
3. K-Means clustering
4. Segment analysis and recommendations
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Import project modules
from src.utils import load_data, clean_data
from src.rfm import calculate_rfm, score_rfm, segment_rfm_by_score, plot_rfm_distributions
from src.clustering import (prepare_rfm_for_clustering, 
                           find_optimal_clusters, 
                           apply_kmeans, 
                           visualize_clusters)

def main():
    """Main execution function"""
    print("=" * 70)
    print("CUSTOMER SEGMENTATION PROJECT")
    print("=" * 70)
    
    # Create directories if they don't exist
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('visualizations', exist_ok=True)
    
    # Step 1: Load and clean data
    print("\n[1/4] Loading and cleaning data...")
    df_raw = load_data('raw_data.csv')
    df_clean = clean_data(df_raw)
    
    # Step 2: RFM Analysis
    print("\n[2/4] Performing RFM analysis...")
    rfm_table = calculate_rfm(df_clean)
    rfm_scored = score_rfm(rfm_table)
    rfm_segmented = segment_rfm_by_score(rfm_scored)
    
    # Plot RFM distributions
    plot_rfm_distributions(rfm_table)
    
    # Export RFM table
    rfm_segmented.to_csv('data/processed/rfm_table.csv', index=False)
    print("  ✓ RFM table exported to data/processed/rfm_table.csv")
    
    # Step 3: Clustering
    print("\n[3/4] Performing customer clustering...")
    scaled_features, scaler, rfm_log = prepare_rfm_for_clustering(rfm_table)
    
    # Find optimal clusters
    inertia_values, optimal_k, fig = find_optimal_clusters(scaled_features, max_k=10)
    print(f"  ✓ Optimal number of clusters: {optimal_k}")
    
    # Apply K-Means
    kmeans_model, cluster_labels, cluster_centers = apply_kmeans(
        scaled_features, 
        n_clusters=optimal_k
    )
    
    # Add cluster labels to RFM table
    rfm_table['cluster'] = cluster_labels
    
    # Visualize clusters
    fig, pca = visualize_clusters(
        pd.DataFrame(scaled_features, columns=['recency', 'frequency', 'monetary']),
        cluster_labels,
        cluster_centers
    )
    
    # Step 4: Create final segments with business names
    print("\n[4/4] Creating business segments...")
    
    # Define business segment names based on cluster analysis
    segment_names = {
        0: 'At-Risk Customers',
        1: 'Promising Customers',
        2: 'Loyal Regulars',
        3: 'High-Value Champions',
        4: 'Seasonal Shoppers'
    }
    
    # Map cluster numbers to names
    rfm_table['segment_name'] = rfm_table['cluster'].map(segment_names)
    
    # Merge with scored RFM data
    final_segments = pd.merge(
        rfm_segmented,
        rfm_table[['customer_id', 'cluster', 'segment_name']],
        on='customer_id',
        how='left'
    )
    
    # Export final segments
    final_segments.to_csv('data/processed/customer_segments.csv', index=False)
    print("  ✓ Customer segments exported to data/processed/customer_segments.csv")
    
    # Generate summary report
    print("\n" + "=" * 70)
    print("PROJECT EXECUTION COMPLETE")
    print("=" * 70)
    
    # Summary statistics
    print(f"\n📊 PROJECT SUMMARY:")
    print(f"   • Customers analyzed: {len(final_segments)}")
    print(f"   • Time period: {df_clean['transaction_date'].min().date()} to {df_clean['transaction_date'].max().date()}")
    print(f"   • Total transactions: {len(df_clean)}")
    print(f"   • Total revenue: ${df_clean['order_value'].sum():,.2f}")
    print(f"   • Segments identified: {len(final_segments['segment_name'].unique())}")
    
    # Segment distribution
    print(f"\n👥 SEGMENT DISTRIBUTION:")
    segment_counts = final_segments['segment_name'].value_counts()
    for segment, count in segment_counts.items():
        percentage = (count / len(final_segments)) * 100
        print(f"   • {segment}: {count} customers ({percentage:.1f}%)")
    
    print(f"\n📁 OUTPUT FILES:")
    print(f"   ✓ data/processed/rfm_table.csv")
    print(f"   ✓ data/processed/customer_segments.csv")
    print(f"   ✓ visualizations/rfm_distributions.png")
    print(f"   ✓ visualizations/elbow_method.png")
    print(f"   ✓ visualizations/cluster_visualization.png")
    
    print(f"\n🎯 NEXT STEPS:")
    print("   1. Review customer_segments.csv for detailed analysis")
    print("   2. Check visualizations/ folder for charts")
    print("   3. Open notebooks/ for step-by-step analysis")
    print("   4. Implement marketing strategies based on segments")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()