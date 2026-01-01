# src/clustering.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from src.utils import save_plot

def prepare_rfm_for_clustering(rfm_table, columns=['recency', 'frequency', 'monetary']):
    """
    Prepare RFM data for clustering with normalization
    
    Parameters:
    -----------
    rfm_table : pandas.DataFrame
        RFM metrics table
    columns : list
        Columns to include in clustering
        
    Returns:
    --------
    tuple
        (scaled_features, scaler, rfm_features)
    """
    # Select RFM features
    rfm_features = rfm_table[columns].copy()
    
    # Apply log transformation to handle skewness (especially for monetary)
    rfm_log = rfm_features.copy()
    for col in ['frequency', 'monetary']:
        if col in rfm_log.columns:
            # Add small constant to handle zeros
            rfm_log[col] = np.log1p(rfm_log[col])
    
    # Scale the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(rfm_log)
    
    return scaled_features, scaler, rfm_log

def find_optimal_clusters(data, max_k=10):
    """
    Find optimal number of clusters using Elbow Method
    
    Parameters:
    -----------
    data : array-like
        Scaled feature data
    max_k : int
        Maximum number of clusters to try
        
    Returns:
    --------
    tuple
        (inertia_values, optimal_k, fig)
    """
    inertia_values = []
    k_range = range(1, max_k + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(data)
        inertia_values.append(kmeans.inertia_)
    
    # Calculate the rate of change of inertia
    differences = np.diff(inertia_values)
    differences_ratio = differences[1:] / differences[:-1]
    
    # Optimal k is where the difference ratio is minimized
    if len(differences_ratio) > 0:
        optimal_k = np.argmin(differences_ratio) + 3  # +3 because we start from k=3
        optimal_k = min(optimal_k, max_k - 1)
    else:
        optimal_k = 3
    
    # Plot elbow curve
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Elbow curve
    ax1.plot(k_range, inertia_values, 'bo-', linewidth=2, markersize=8)
    ax1.axvline(x=optimal_k, color='r', linestyle='--', alpha=0.7)
    ax1.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax1.set_ylabel('Inertia', fontsize=12)
    ax1.set_title('Elbow Method for Optimal k', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Rate of change
    ax2.plot(range(2, max_k + 1), differences, 'go-', linewidth=2, markersize=8)
    ax2.axvline(x=optimal_k, color='r', linestyle='--', alpha=0.7)
    ax2.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax2.set_ylabel('Inertia Difference', fontsize=12)
    ax2.set_title('Rate of Change in Inertia', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_plot(fig, 'elbow_method.png')
    
    return inertia_values, optimal_k, fig

def apply_kmeans(data, n_clusters, random_state=42):
    """
    Apply K-Means clustering
    
    Parameters:
    -----------
    data : array-like
        Scaled feature data
    n_clusters : int
        Number of clusters
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    tuple
        (kmeans_model, cluster_labels, cluster_centers)
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(data)
    cluster_centers = kmeans.cluster_centers_
    
    # Calculate silhouette score
    if n_clusters > 1:
        silhouette_avg = silhouette_score(data, cluster_labels)
        print(f"Silhouette Score for {n_clusters} clusters: {silhouette_avg:.3f}")
    
    return kmeans, cluster_labels, cluster_centers

def visualize_clusters(rfm_features, cluster_labels, cluster_centers=None):
    """
    Visualize clusters using PCA for dimensionality reduction
    
    Parameters:
    -----------
    rfm_features : pandas.DataFrame
        Original RFM features (scaled or unscaled)
    cluster_labels : array-like
        Cluster assignments
    cluster_centers : array-like, optional
        Cluster centers
        
    Returns:
    --------
    matplotlib.figure.Figure
        Cluster visualization figure
    """
    # Apply PCA for 2D visualization
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(rfm_features)
    
    # Create DataFrame for plotting
    plot_df = pd.DataFrame({
        'PC1': pca_result[:, 0],
        'PC2': pca_result[:, 1],
        'Cluster': cluster_labels
    })
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Scatter plot colored by cluster
    scatter = ax1.scatter(plot_df['PC1'], plot_df['PC2'], 
                         c=plot_df['Cluster'], cmap='tab10', 
                         alpha=0.7, s=50, edgecolor='k', linewidth=0.5)
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
    ax1.set_title('Customer Clusters (PCA Reduced)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add cluster centers if provided
    if cluster_centers is not None:
        pca_centers = pca.transform(cluster_centers)
        ax1.scatter(pca_centers[:, 0], pca_centers[:, 1], 
                   c='red', marker='X', s=200, label='Cluster Centers',
                   edgecolor='k', linewidth=1.5)
        ax1.legend()
    
    # Box plot of clusters by RFM metrics
    cluster_data = []
    for col in rfm_features.columns:
        cluster_data.append(pd.DataFrame({
            'value': rfm_features[col],
            'cluster': cluster_labels,
            'metric': col
        }))
    
    cluster_df = pd.concat(cluster_data)
    
    # Create boxplot
    sns.boxplot(data=cluster_df, x='cluster', y='value', hue='metric', 
                ax=ax2, palette='Set2')
    ax2.set_xlabel('Cluster', fontsize=12)
    ax2.set_ylabel('Scaled Value', fontsize=12)
    ax2.set_title('RFM Metrics by Cluster', fontsize=14, fontweight='bold')
    ax2.legend(title='RFM Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_plot(fig, 'cluster_visualization.png')
    
    return fig, pca