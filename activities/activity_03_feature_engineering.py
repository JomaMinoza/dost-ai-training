"""
Activity 3: Feature Engineering & Selection

DOST-ITDI AI Training Workshop
Learning Objectives:
1. Understand feature correlation and multicollinearity
2. Apply feature selection techniques (RFE, SelectKBest)
3. Create polynomial features for non-linear relationships
4. Compare feature importance across models
5. Optimize feature sets for better model performance
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import (
    RFE, SelectKBest, f_regression, mutual_info_regression
)
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("Activity 3: Feature Engineering & Selection")
print("="*70)

# Load dataset
url = "https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/delaney-processed.csv"
df = pd.read_csv(url)

# Calculate additional descriptors
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
            'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
            'NumAromaticRings': Descriptors.NumAromaticRings(mol),
            'FractionCSP3': Descriptors.FractionCSP3(mol),
            'NumRings': Descriptors.RingCount(mol),
            'NumSaturatedRings': Descriptors.NumSaturatedRings(mol)
        }
    except:
        return None

descriptors_list = [calculate_descriptors(smiles) for smiles in df['smiles']]
descriptors_df = pd.DataFrame(descriptors_list).dropna()

# Merge with existing features
df_clean = df.loc[descriptors_df.index].copy()
for col in descriptors_df.columns:
    df_clean[col] = descriptors_df[col].values

# Select all feature columns
feature_columns = [
    'Molecular Weight', 'Number of H-Bond Donors', 'Number of Rings',
    'Number of Rotatable Bonds', 'Polar Surface Area',
    'MolWeight', 'LogP', 'NumHDonors', 'NumHAcceptors', 'TPSA',
    'NumRotatableBonds', 'NumAromaticRings', 'FractionCSP3',
    'NumRings', 'NumSaturatedRings'
]

X = df_clean[feature_columns].values
y = df_clean['measured log solubility in mols per litre'].values

print(f"\nDataset: {X.shape}")
print(f"Features: {len(feature_columns)}")

# ============================================================================
# 1. FEATURE CORRELATION ANALYSIS
# ============================================================================
print("\n" + "="*70)
print("1. FEATURE CORRELATION ANALYSIS")
print("="*70)

# Create correlation matrix
correlation_matrix = pd.DataFrame(X, columns=feature_columns).corr()

# Plot correlation heatmap
plt.figure(figsize=(14, 12))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('feature_correlation.png', dpi=150, bbox_inches='tight')
print("\nCorrelation heatmap saved to 'feature_correlation.png'")

# Find highly correlated feature pairs
print("\nHighly Correlated Feature Pairs (|r| > 0.9):")
high_corr_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        if abs(correlation_matrix.iloc[i, j]) > 0.9:
            high_corr_pairs.append({
                'Feature 1': correlation_matrix.columns[i],
                'Feature 2': correlation_matrix.columns[j],
                'Correlation': correlation_matrix.iloc[i, j]
            })

if high_corr_pairs:
    for pair in high_corr_pairs:
        print(f"  {pair['Feature 1']:30s} <-> {pair['Feature 2']:30s}: {pair['Correlation']:.3f}")
    print(f"\n⚠ Warning: High correlation may indicate multicollinearity!")
    print("  Consider removing one feature from each pair.")
else:
    print("  No highly correlated pairs found.")

# ============================================================================
# 2. FEATURE IMPORTANCE - MULTIPLE METHODS
# ============================================================================
print("\n" + "="*70)
print("2. FEATURE IMPORTANCE - MULTIPLE METHODS")
print("="*70)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Method 1: Random Forest Feature Importance
print("\nMethod 1: Random Forest Feature Importance")
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train_scaled, y_train)
rf_importance = pd.DataFrame({
    'Feature': feature_columns,
    'RF_Importance': rf.feature_importances_
}).sort_values('RF_Importance', ascending=False)

print("Top 10 features (Random Forest):")
print(rf_importance.head(10).to_string(index=False))

# Method 2: Gradient Boosting Feature Importance
print("\nMethod 2: Gradient Boosting Feature Importance")
gbr = GradientBoostingRegressor(n_estimators=100, random_state=42)
gbr.fit(X_train_scaled, y_train)
gb_importance = pd.DataFrame({
    'Feature': feature_columns,
    'GB_Importance': gbr.feature_importances_
}).sort_values('GB_Importance', ascending=False)

print("Top 10 features (Gradient Boosting):")
print(gb_importance.head(10).to_string(index=False))

# Method 3: L1 (Lasso) Coefficients
print("\nMethod 3: Lasso (L1) Coefficient Magnitude")
lasso = Lasso(alpha=0.01, random_state=42)
lasso.fit(X_train_scaled, y_train)
lasso_importance = pd.DataFrame({
    'Feature': feature_columns,
    'Lasso_Coef': np.abs(lasso.coef_)
}).sort_values('Lasso_Coef', ascending=False)

print("Top 10 features (Lasso coefficients):")
print(lasso_importance.head(10).to_string(index=False))

# Visualize comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Random Forest
top_10_rf = rf_importance.head(10)
axes[0].barh(top_10_rf['Feature'], top_10_rf['RF_Importance'],
             color='steelblue', edgecolor='black')
axes[0].set_xlabel('Importance', fontsize=11)
axes[0].set_title('Random Forest\nFeature Importance', fontsize=12, fontweight='bold')
axes[0].invert_yaxis()
axes[0].grid(True, alpha=0.3, axis='x')

# Gradient Boosting
top_10_gb = gb_importance.head(10)
axes[1].barh(top_10_gb['Feature'], top_10_gb['GB_Importance'],
             color='coral', edgecolor='black')
axes[1].set_xlabel('Importance', fontsize=11)
axes[1].set_title('Gradient Boosting\nFeature Importance', fontsize=12, fontweight='bold')
axes[1].invert_yaxis()
axes[1].grid(True, alpha=0.3, axis='x')

# Lasso
top_10_lasso = lasso_importance.head(10)
axes[2].barh(top_10_lasso['Feature'], top_10_lasso['Lasso_Coef'],
             color='seagreen', edgecolor='black')
axes[2].set_xlabel('Absolute Coefficient', fontsize=11)
axes[2].set_title('Lasso\nCoefficient Magnitude', fontsize=12, fontweight='bold')
axes[2].invert_yaxis()
axes[2].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('feature_importance_comparison.png', dpi=150, bbox_inches='tight')
print("\nFeature importance comparison saved to 'feature_importance_comparison.png'")

# ============================================================================
# 3. UNIVARIATE FEATURE SELECTION
# ============================================================================
print("\n" + "="*70)
print("3. UNIVARIATE FEATURE SELECTION")
print("="*70)

# F-statistic
print("\nMethod 1: F-statistic (SelectKBest)")
selector_f = SelectKBest(score_func=f_regression, k=10)
selector_f.fit(X_train_scaled, y_train)

f_scores = pd.DataFrame({
    'Feature': feature_columns,
    'F_Score': selector_f.scores_
}).sort_values('F_Score', ascending=False)

print("Top 10 features by F-statistic:")
print(f_scores.head(10).to_string(index=False))

# Mutual Information
print("\nMethod 2: Mutual Information")
selector_mi = SelectKBest(score_func=mutual_info_regression, k=10)
selector_mi.fit(X_train_scaled, y_train)

mi_scores = pd.DataFrame({
    'Feature': feature_columns,
    'MI_Score': selector_mi.scores_
}).sort_values('MI_Score', ascending=False)

print("Top 10 features by Mutual Information:")
print(mi_scores.head(10).to_string(index=False))

# ============================================================================
# 4. RECURSIVE FEATURE ELIMINATION (RFE)
# ============================================================================
print("\n" + "="*70)
print("4. RECURSIVE FEATURE ELIMINATION (RFE)")
print("="*70)

estimator = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)

# Try different numbers of features
n_features_to_select_list = [5, 8, 10, 12]
rfe_results = []

for n_features in n_features_to_select_list:
    rfe = RFE(estimator=estimator, n_features_to_select=n_features)
    rfe.fit(X_train_scaled, y_train)

    # Get selected features
    selected_features = [feature_columns[i] for i in range(len(feature_columns)) if rfe.support_[i]]

    # Train model with selected features
    X_train_rfe = rfe.transform(X_train_scaled)
    X_test_rfe = rfe.transform(X_test_scaled)

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_rfe, y_train)

    y_pred = model.predict(X_test_rfe)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    rfe_results.append({
        'N_Features': n_features,
        'R2_Score': r2,
        'RMSE': rmse,
        'Selected_Features': selected_features
    })

    print(f"\nRFE with {n_features} features:")
    print(f"  R² Score: {r2:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  Selected: {', '.join(selected_features[:5])}...")

# Plot RFE results
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

n_features = [r['N_Features'] for r in rfe_results]
r2_scores = [r['R2_Score'] for r in rfe_results]
rmse_scores = [r['RMSE'] for r in rfe_results]

axes[0].plot(n_features, r2_scores, 'o-', linewidth=2, markersize=10, color='steelblue')
axes[0].set_xlabel('Number of Features', fontsize=12)
axes[0].set_ylabel('R² Score', fontsize=12)
axes[0].set_title('RFE: Number of Features vs R²', fontsize=13, fontweight='bold')
axes[0].grid(True, alpha=0.3)

axes[1].plot(n_features, rmse_scores, 's-', linewidth=2, markersize=10, color='coral')
axes[1].set_xlabel('Number of Features', fontsize=12)
axes[1].set_ylabel('RMSE', fontsize=12)
axes[1].set_title('RFE: Number of Features vs RMSE', fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('rfe_results.png', dpi=150, bbox_inches='tight')
print("\nRFE results saved to 'rfe_results.png'")

# ============================================================================
# 5. POLYNOMIAL FEATURES
# ============================================================================
print("\n" + "="*70)
print("5. POLYNOMIAL FEATURES FOR NON-LINEAR RELATIONSHIPS")
print("="*70)

# Select a subset of features to avoid explosion
important_features = ['LogP', 'TPSA', 'MolWeight', 'NumHDonors', 'NumHAcceptors']
feature_indices = [feature_columns.index(f) for f in important_features]

X_subset = X[:, feature_indices]
X_train_sub, X_test_sub, y_train_sub, y_test_sub = train_test_split(
    X_subset, y, test_size=0.2, random_state=42
)

# Scale
scaler_sub = StandardScaler()
X_train_sub_scaled = scaler_sub.fit_transform(X_train_sub)
X_test_sub_scaled = scaler_sub.transform(X_test_sub)

# Compare different polynomial degrees
print("\nComparing polynomial degrees:")
poly_results = []

for degree in [1, 2]:  # degree=1 is baseline (linear)
    if degree == 1:
        X_train_poly = X_train_sub_scaled
        X_test_poly = X_test_sub_scaled
        n_features = X_train_poly.shape[1]
    else:
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_train_poly = poly.fit_transform(X_train_sub_scaled)
        X_test_poly = poly.transform(X_test_sub_scaled)
        n_features = X_train_poly.shape[1]

    # Train model
    model = Ridge(alpha=1.0, random_state=42)
    model.fit(X_train_poly, y_train_sub)

    y_pred_train = model.predict(X_train_poly)
    y_pred_test = model.predict(X_test_poly)

    train_r2 = r2_score(y_train_sub, y_pred_train)
    test_r2 = r2_score(y_test_sub, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test_sub, y_pred_test))

    poly_results.append({
        'Degree': degree,
        'N_Features': n_features,
        'Train_R2': train_r2,
        'Test_R2': test_r2,
        'RMSE': test_rmse
    })

    print(f"\nPolynomial Degree {degree}:")
    print(f"  Features: {important_features} -> {n_features} features")
    print(f"  Train R²: {train_r2:.4f}")
    print(f"  Test R²: {test_r2:.4f}")
    print(f"  Test RMSE: {test_rmse:.4f}")

poly_df = pd.DataFrame(poly_results)
print("\nPolynomial Features Summary:")
print(poly_df.to_string(index=False))

# ============================================================================
# 6. FEATURE SELECTION PIPELINE COMPARISON
# ============================================================================
print("\n" + "="*70)
print("6. COMPARING FEATURE SELECTION STRATEGIES")
print("="*70)

strategies = {
    'All Features': list(range(len(feature_columns))),
    'Top 10 RF': rf_importance.head(10)['Feature'].tolist(),
    'Top 10 F-test': f_scores.head(10)['Feature'].tolist(),
    'Top 10 MI': mi_scores.head(10)['Feature'].tolist(),
}

comparison_results = []

for strategy_name, selected in strategies.items():
    # Get feature indices
    if isinstance(selected, list) and isinstance(selected[0], str):
        indices = [feature_columns.index(f) for f in selected]
    else:
        indices = selected

    X_selected = X[:, indices]
    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
        X_selected, y, test_size=0.2, random_state=42
    )

    scaler_s = StandardScaler()
    X_train_s_scaled = scaler_s.fit_transform(X_train_s)
    X_test_s_scaled = scaler_s.transform(X_test_s)

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_s_scaled, y_train_s)

    y_pred = model.predict(X_test_s_scaled)
    r2 = r2_score(y_test_s, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_s, y_pred))

    comparison_results.append({
        'Strategy': strategy_name,
        'N_Features': len(indices),
        'Test_R2': r2,
        'RMSE': rmse
    })

    print(f"\n{strategy_name}:")
    print(f"  Features: {len(indices)}")
    print(f"  Test R²: {r2:.4f}")
    print(f"  RMSE: {rmse:.4f}")

comparison_df = pd.DataFrame(comparison_results)
print("\nStrategy Comparison:")
print(comparison_df.to_string(index=False))

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].barh(comparison_df['Strategy'], comparison_df['Test_R2'],
             color='steelblue', edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Test R² Score', fontsize=12)
axes[0].set_title('Feature Selection Strategy Comparison (R²)',
                  fontsize=13, fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='x')

axes[1].barh(comparison_df['Strategy'], comparison_df['RMSE'],
             color='coral', edgecolor='black', alpha=0.7)
axes[1].set_xlabel('Test RMSE', fontsize=12)
axes[1].set_title('Feature Selection Strategy Comparison (RMSE)',
                  fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('strategy_comparison.png', dpi=150, bbox_inches='tight')
print("\nStrategy comparison saved to 'strategy_comparison.png'")

# ============================================================================
# 7. KEY RECOMMENDATIONS
# ============================================================================
print("\n" + "="*70)
print("7. FEATURE ENGINEERING BEST PRACTICES")
print("="*70)

print("""
FEATURE ENGINEERING WORKFLOW:

1. START WITH DOMAIN KNOWLEDGE:
   ✅ Understand what each feature means
   ✅ Know which features are physically/chemically meaningful
   ✅ Consult domain experts

2. EXPLORE CORRELATIONS:
   ✅ Check correlation matrix
   ✅ Remove redundant features (|r| > 0.95)
   ✅ Watch for multicollinearity in linear models

3. FEATURE IMPORTANCE ANALYSIS:
   ✅ Use multiple methods (RF, GB, Lasso)
   ✅ Look for consensus across methods
   ✅ Keep features important in multiple models

4. FEATURE SELECTION TECHNIQUES:
   ✅ Univariate: F-test, Mutual Information (fast, independent)
   ✅ RFE: Iterative removal (slower, model-dependent)
   ✅ L1 Regularization: Automatic via Lasso
   ✅ Tree-based: Built-in importance

5. CREATE NEW FEATURES:
   ✅ Polynomial features for non-linear relationships
   ✅ Interaction terms for synergistic effects
   ✅ Domain-specific transformations (log, sqrt, etc.)
   ✅ Binning/discretization for categorical behavior

6. AVOID COMMON PITFALLS:
   ❌ Don't select features on the entire dataset (data leakage!)
   ❌ Don't create too many polynomial features (curse of dimensionality)
   ❌ Don't remove features without understanding why
   ❌ Don't forget to scale features for distance-based models

7. VALIDATE SELECTION:
   ✅ Use cross-validation to assess stability
   ✅ Check performance on held-out test set
   ✅ Monitor for overfitting (train vs test gap)
   ✅ Consider computational cost vs performance gain

WHEN TO USE EACH METHOD:

SelectKBest (F-test):
  → Fast, univariate, good starting point
  → Best for: Linear relationships, quick screening

Mutual Information:
  → Captures non-linear relationships
  → Best for: Non-linear models, complex patterns

RFE:
  → Considers feature interactions
  → Best for: When you have time, important projects

L1 (Lasso):
  → Automatic, built into training
  → Best for: Linear models, automatic pipelines

Tree-based Importance:
  → Fast, handles interactions
  → Best for: Tree/ensemble models, non-linear

CHEMISTRY-SPECIFIC TIPS:

1. Lipinski's Rule of Five features are often important:
   - Molecular Weight < 500
   - LogP < 5
   - H-bond donors < 5
   - H-bond acceptors < 10

2. Common important descriptors:
   - LogP (lipophilicity)
   - TPSA (polar surface area)
   - Number of aromatic rings
   - Number of rotatable bonds

3. Consider creating:
   - Ratios (e.g., heteroatoms/total atoms)
   - Aromatic vs aliphatic features
   - Ring strain indicators
   - Functional group presence

EXERCISE FOR YOU:
1. Try different feature selection k values (k=5, 10, 15)
2. Create custom features based on chemistry knowledge
3. Use PCA for dimensionality reduction
4. Compare feature selection impact on different models
5. Implement forward/backward selection
""")

print("\n" + "="*70)
print("Activity 3 Complete!")
print("="*70)
