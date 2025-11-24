"""
Activity 7: Clustering (Unsupervised Learning)

DOST-ITDI AI Training Workshop
Learning Objectives:
1. Understand unsupervised learning concepts
2. Apply K-Means clustering
3. Use Hierarchical clustering
4. Apply DBSCAN for density-based clustering
5. Evaluate clustering quality
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("Activity 7: Clustering (Unsupervised Learning)")
print("="*70)

# Load dataset
url = "https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/delaney-processed.csv"
df = pd.read_csv(url)

from rdkit import Chem
from rdkit.Chem import Descriptors

def calculate_descriptors(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return {
            'MolWeight': Descriptors.MolWt(mol),
            'LogP': Descriptors.MolLogP(mol),
            'NumHDonors': Descriptors.NumHDonors(mol),
            'NumHAcceptors': Descriptors.NumHAcceptors(mol),
            'TPSA': Descriptors.TPSA(mol),
            'NumRotatableBonds': Descriptors.NumRotatableBonds(mol)
        }
    except:
        return None

descriptors_list = [calculate_descriptors(smiles) for smiles in df['smiles']]
descriptors_df = pd.DataFrame(descriptors_list).dropna()

feature_columns = list(descriptors_df.columns)
X = descriptors_df.values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"\nDataset: {X_scaled.shape[0]} molecules, {X_scaled.shape[1]} features")
print(f"Features: {feature_columns}")

# ============================================================================
# 1. K-MEANS CLUSTERING
# ============================================================================
print("\n" + "="*70)
print("1. K-MEANS CLUSTERING")
print("="*70)

# Elbow method to find optimal K
inertias = []
silhouettes = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouettes.append(silhouette_score(X_scaled, kmeans.labels_))

# Plot elbow and silhouette
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(K_range, inertias, 'o-', linewidth=2, markersize=8)
axes[0].set_xlabel('Number of Clusters (K)', fontsize=12)
axes[0].set_ylabel('Inertia', fontsize=12)
axes[0].set_title('Elbow Method', fontsize=13, fontweight='bold')
axes[0].grid(True, alpha=0.3)

axes[1].plot(K_range, silhouettes, 's-', linewidth=2, markersize=8, color='green')
axes[1].set_xlabel('Number of Clusters (K)', fontsize=12)
axes[1].set_ylabel('Silhouette Score', fontsize=12)
axes[1].set_title('Silhouette Analysis', fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('kmeans_elbow.png', dpi=150, bbox_inches='tight')
print("\nElbow plot saved to 'kmeans_elbow.png'")

# Optimal K
optimal_k = K_range[np.argmax(silhouettes)]
print(f"\nOptimal K (by silhouette): {optimal_k}")

# Final K-Means
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)

print(f"\nCluster sizes:")
for i in range(optimal_k):
    print(f"  Cluster {i}: {np.sum(kmeans_labels == i)} molecules")

# ============================================================================
# 2. HIERARCHICAL CLUSTERING
# ============================================================================
print("\n" + "="*70)
print("2. HIERARCHICAL CLUSTERING")
print("="*70)

# Dendrogram (use subset for visibility)
sample_size = min(100, len(X_scaled))
X_sample = X_scaled[:sample_size]

plt.figure(figsize=(14, 6))
linkage_matrix = linkage(X_sample, method='ward')
dendrogram(linkage_matrix, truncate_mode='level', p=5)
plt.xlabel('Sample Index', fontsize=12)
plt.ylabel('Distance', fontsize=12)
plt.title('Hierarchical Clustering Dendrogram', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('dendrogram.png', dpi=150, bbox_inches='tight')
print("\nDendrogram saved to 'dendrogram.png'")

# Agglomerative clustering
agg = AgglomerativeClustering(n_clusters=optimal_k)
agg_labels = agg.fit_predict(X_scaled)

print(f"\nAgglomerative Clustering (K={optimal_k}):")
for i in range(optimal_k):
    print(f"  Cluster {i}: {np.sum(agg_labels == i)} molecules")

# ============================================================================
# 3. DBSCAN (Density-Based)
# ============================================================================
print("\n" + "="*70)
print("3. DBSCAN (Density-Based Clustering)")
print("="*70)

dbscan = DBSCAN(eps=1.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)

n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise = np.sum(dbscan_labels == -1)

print(f"\nDBSCAN Results:")
print(f"  Clusters found: {n_clusters_dbscan}")
print(f"  Noise points: {n_noise}")

for i in range(n_clusters_dbscan):
    print(f"  Cluster {i}: {np.sum(dbscan_labels == i)} molecules")

# ============================================================================
# 4. VISUALIZATION (PCA)
# ============================================================================
print("\n" + "="*70)
print("4. CLUSTER VISUALIZATION")
print("="*70)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# K-Means
scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels,
                           cmap='viridis', alpha=0.6, s=30)
axes[0].set_xlabel('PC1', fontsize=11)
axes[0].set_ylabel('PC2', fontsize=11)
axes[0].set_title(f'K-Means (K={optimal_k})', fontsize=12, fontweight='bold')
plt.colorbar(scatter1, ax=axes[0])

# Hierarchical
scatter2 = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=agg_labels,
                           cmap='viridis', alpha=0.6, s=30)
axes[1].set_xlabel('PC1', fontsize=11)
axes[1].set_ylabel('PC2', fontsize=11)
axes[1].set_title(f'Hierarchical (K={optimal_k})', fontsize=12, fontweight='bold')
plt.colorbar(scatter2, ax=axes[1])

# DBSCAN
scatter3 = axes[2].scatter(X_pca[:, 0], X_pca[:, 1], c=dbscan_labels,
                           cmap='viridis', alpha=0.6, s=30)
axes[2].set_xlabel('PC1', fontsize=11)
axes[2].set_ylabel('PC2', fontsize=11)
axes[2].set_title(f'DBSCAN ({n_clusters_dbscan} clusters)', fontsize=12, fontweight='bold')
plt.colorbar(scatter3, ax=axes[2])

plt.tight_layout()
plt.savefig('clustering_comparison.png', dpi=150, bbox_inches='tight')
print("\nCluster visualization saved to 'clustering_comparison.png'")

# ============================================================================
# 5. CLUSTER EVALUATION
# ============================================================================
print("\n" + "="*70)
print("5. CLUSTER EVALUATION METRICS")
print("="*70)

results = []

# K-Means
results.append({
    'Method': 'K-Means',
    'N_Clusters': optimal_k,
    'Silhouette': silhouette_score(X_scaled, kmeans_labels),
    'Calinski-Harabasz': calinski_harabasz_score(X_scaled, kmeans_labels)
})

# Hierarchical
results.append({
    'Method': 'Hierarchical',
    'N_Clusters': optimal_k,
    'Silhouette': silhouette_score(X_scaled, agg_labels),
    'Calinski-Harabasz': calinski_harabasz_score(X_scaled, agg_labels)
})

# DBSCAN (exclude noise for evaluation)
if n_clusters_dbscan > 1:
    mask = dbscan_labels != -1
    results.append({
        'Method': 'DBSCAN',
        'N_Clusters': n_clusters_dbscan,
        'Silhouette': silhouette_score(X_scaled[mask], dbscan_labels[mask]),
        'Calinski-Harabasz': calinski_harabasz_score(X_scaled[mask], dbscan_labels[mask])
    })

results_df = pd.DataFrame(results)
print("\nClustering Comparison:")
print(results_df.to_string(index=False))

# ============================================================================
# 6. CLUSTER PROFILING
# ============================================================================
print("\n" + "="*70)
print("6. CLUSTER PROFILING (K-Means)")
print("="*70)

descriptors_df['Cluster'] = kmeans_labels

print("\nCluster Centers (Mean values):")
cluster_profile = descriptors_df.groupby('Cluster')[feature_columns].mean()
print(cluster_profile.round(2).to_string())

# ============================================================================
# 7. SUMMARY
# ============================================================================
print("\n" + "="*70)
print("7. CLUSTERING METHODS SUMMARY")
print("="*70)

print("""
K-MEANS:
  + Fast, scalable
  + Works well with spherical clusters
  - Requires specifying K
  - Sensitive to outliers

HIERARCHICAL:
  + No need to specify K upfront
  + Dendrogram shows structure
  - Slow for large datasets
  - Cannot undo merges

DBSCAN:
  + Finds arbitrary shapes
  + Detects outliers automatically
  - Sensitive to eps and min_samples
  - Struggles with varying densities

WHEN TO USE:
  K-Means: Large data, spherical clusters
  Hierarchical: Small data, need hierarchy
  DBSCAN: Unknown K, noisy data, arbitrary shapes
""")

print("\n" + "="*70)
print("Activity 7 Complete!")
print("="*70)
