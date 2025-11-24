"""
Activity 4: Model Interpretation & Explainability

DOST-ITDI AI Training Workshop
Learning Objectives:
1. Understand permutation importance for any model
2. Analyze partial dependence plots
3. Visualize individual predictions
4. Compare interpretation methods
5. Apply explainability to chemistry problems
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.inspection import permutation_importance, partial_dependence, PartialDependenceDisplay
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("Activity 4: Model Interpretation & Explainability")
print("="*70)

# Load dataset
url = "https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/delaney-processed.csv"
df = pd.read_csv(url)

# Calculate descriptors
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

df_clean = df.loc[descriptors_df.index].copy()

feature_columns = [
    'Molecular Weight',
    'Number of H-Bond Donors',
    'Number of Rings',
    'Number of Rotatable Bonds',
    'Polar Surface Area',
    'LogP',
    'NumHAcceptors',
    'TPSA'
]

# Add calculated features
for col in descriptors_df.columns:
    if col not in df_clean.columns:
        df_clean[col] = descriptors_df[col].values

X = df_clean[feature_columns].values
y = df_clean['measured log solubility in mols per litre'].values

print(f"\nDataset: {X.shape}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train models
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train_scaled, y_train)

gbr = GradientBoostingRegressor(n_estimators=100, random_state=42)
gbr.fit(X_train_scaled, y_train)

# Evaluate
y_pred_rf = rf.predict(X_test_scaled)
y_pred_gbr = gbr.predict(X_test_scaled)

print(f"\nRandom Forest - Test R^2: {r2_score(y_test, y_pred_rf):.4f}")
print(f"Gradient Boosting - Test R^2: {r2_score(y_test, y_pred_gbr):.4f}")

# ============================================================================
# 1. FEATURE IMPORTANCE (Built-in)
# ============================================================================
print("\n" + "="*70)
print("1. BUILT-IN FEATURE IMPORTANCE")
print("="*70)

rf_importance = pd.DataFrame({
    'Feature': feature_columns,
    'RF_Importance': rf.feature_importances_
}).sort_values('RF_Importance', ascending=False)

gb_importance = pd.DataFrame({
    'Feature': feature_columns,
    'GB_Importance': gbr.feature_importances_
}).sort_values('GB_Importance', ascending=False)

print("\nRandom Forest Feature Importance:")
print(rf_importance.to_string(index=False))

print("\nGradient Boosting Feature Importance:")
print(gb_importance.to_string(index=False))

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].barh(rf_importance['Feature'], rf_importance['RF_Importance'],
             color='steelblue', edgecolor='black')
axes[0].set_xlabel('Importance', fontsize=12)
axes[0].set_title('Random Forest\nFeature Importance', fontsize=13, fontweight='bold')
axes[0].invert_yaxis()
axes[0].grid(True, alpha=0.3, axis='x')

axes[1].barh(gb_importance['Feature'], gb_importance['GB_Importance'],
             color='coral', edgecolor='black')
axes[1].set_xlabel('Importance', fontsize=12)
axes[1].set_title('Gradient Boosting\nFeature Importance', fontsize=13, fontweight='bold')
axes[1].invert_yaxis()
axes[1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('builtin_feature_importance.png', dpi=150, bbox_inches='tight')
print("\nFeature importance plot saved to 'builtin_feature_importance.png'")

# ============================================================================
# 2. PERMUTATION IMPORTANCE
# ============================================================================
print("\n" + "="*70)
print("2. PERMUTATION IMPORTANCE")
print("="*70)

print("\nPermutation importance works by shuffling one feature at a time")
print("and measuring the drop in model performance.")
print("Higher values = more important features")

# Calculate permutation importance
perm_importance_rf = permutation_importance(
    rf, X_test_scaled, y_test, n_repeats=10, random_state=42, n_jobs=-1
)

perm_importance_gbr = permutation_importance(
    gbr, X_test_scaled, y_test, n_repeats=10, random_state=42, n_jobs=-1
)

# Create DataFrames
perm_rf_df = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': perm_importance_rf.importances_mean,
    'Std': perm_importance_rf.importances_std
}).sort_values('Importance', ascending=False)

perm_gbr_df = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': perm_importance_gbr.importances_mean,
    'Std': perm_importance_gbr.importances_std
}).sort_values('Importance', ascending=False)

print("\nRandom Forest Permutation Importance:")
print(perm_rf_df.to_string(index=False))

print("\nGradient Boosting Permutation Importance:")
print(perm_gbr_df.to_string(index=False))

# Visualize with error bars
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].barh(perm_rf_df['Feature'], perm_rf_df['Importance'],
             xerr=perm_rf_df['Std'], color='steelblue',
             edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Importance (Drop in R^2)', fontsize=12)
axes[0].set_title('Random Forest\nPermutation Importance', fontsize=13, fontweight='bold')
axes[0].invert_yaxis()
axes[0].grid(True, alpha=0.3, axis='x')

axes[1].barh(perm_gbr_df['Feature'], perm_gbr_df['Importance'],
             xerr=perm_gbr_df['Std'], color='coral',
             edgecolor='black', alpha=0.7)
axes[1].set_xlabel('Importance (Drop in R^2)', fontsize=12)
axes[1].set_title('Gradient Boosting\nPermutation Importance', fontsize=13, fontweight='bold')
axes[1].invert_yaxis()
axes[1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('permutation_importance.png', dpi=150, bbox_inches='tight')
print("\nPermutation importance plot saved to 'permutation_importance.png'")

# ============================================================================
# 3. PARTIAL DEPENDENCE PLOTS
# ============================================================================
print("\n" + "="*70)
print("3. PARTIAL DEPENDENCE PLOTS")
print("="*70)

print("\nPartial Dependence Plots show the marginal effect of a feature")
print("on the predicted outcome, averaging out all other features.")

# Select top 4 important features
top_features = rf_importance.head(4)['Feature'].tolist()
top_feature_indices = [feature_columns.index(f) for f in top_features]

print(f"\nAnalyzing top 4 features: {', '.join(top_features)}")

# Create PDP for Random Forest
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()

for idx, (feature_name, feature_idx) in enumerate(zip(top_features, top_feature_indices)):
    # Calculate partial dependence
    pdp_result = partial_dependence(
        rf, X_train_scaled, [feature_idx], grid_resolution=50
    )

    # Plot
    axes[idx].plot(pdp_result['grid_values'][0], pdp_result['average'][0],
                   linewidth=3, color='steelblue')
    axes[idx].set_xlabel(feature_name, fontsize=11)
    axes[idx].set_ylabel('Partial Dependence', fontsize=11)
    axes[idx].set_title(f'PDP: {feature_name}', fontsize=12, fontweight='bold')
    axes[idx].grid(True, alpha=0.3)

    # Add interpretation
    y_vals = pdp_result['average'][0]
    trend = "increasing" if y_vals[-1] > y_vals[0] else "decreasing"
    axes[idx].text(0.05, 0.95, f'Trend: {trend}',
                   transform=axes[idx].transAxes,
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

plt.tight_layout()
plt.savefig('partial_dependence_plots.png', dpi=150, bbox_inches='tight')
print("\nPartial dependence plots saved to 'partial_dependence_plots.png'")

# ============================================================================
# 4. TWO-WAY PARTIAL DEPENDENCE (INTERACTION)
# ============================================================================
print("\n" + "="*70)
print("4. TWO-WAY PARTIAL DEPENDENCE (FEATURE INTERACTIONS)")
print("="*70)

print("\nTwo-way PDP shows how two features interact to affect predictions.")

# Select top 2 features for interaction
feature1_name, feature2_name = top_features[0], top_features[1]
feature1_idx = feature_columns.index(feature1_name)
feature2_idx = feature_columns.index(feature2_name)

print(f"\nAnalyzing interaction: {feature1_name} x {feature2_name}")

# Create 2D PDP
fig, ax = plt.subplots(figsize=(10, 8))

pdp_2d = partial_dependence(
    rf, X_train_scaled, [(feature1_idx, feature2_idx)],
    grid_resolution=20
)

# Create heatmap
XX, YY = np.meshgrid(pdp_2d['grid_values'][0], pdp_2d['grid_values'][1])
Z = pdp_2d['average'][0].T

im = ax.contourf(XX, YY, Z, levels=15, cmap='RdYlBu_r', alpha=0.8)
ax.contour(XX, YY, Z, levels=10, colors='black', alpha=0.3, linewidths=0.5)

ax.set_xlabel(feature1_name, fontsize=12)
ax.set_ylabel(feature2_name, fontsize=12)
ax.set_title(f'2D Partial Dependence: {feature1_name} x {feature2_name}',
             fontsize=14, fontweight='bold')

plt.colorbar(im, ax=ax, label='Partial Dependence')
plt.tight_layout()
plt.savefig('partial_dependence_2d.png', dpi=150, bbox_inches='tight')
print("\n2D partial dependence plot saved to 'partial_dependence_2d.png'")

# ============================================================================
# 5. INDIVIDUAL PREDICTION EXPLANATION
# ============================================================================
print("\n" + "="*70)
print("5. INDIVIDUAL PREDICTION EXPLANATION")
print("="*70)

print("\nExplaining specific predictions by showing feature contributions.")

# Select a test sample
sample_idx = 0
sample = X_test_scaled[sample_idx:sample_idx+1]
actual_value = y_test[sample_idx]
predicted_value = rf.predict(sample)[0]

print(f"\nSample {sample_idx}:")
print(f"  Actual solubility: {actual_value:.4f}")
print(f"  Predicted solubility: {predicted_value:.4f}")
print(f"  Error: {abs(actual_value - predicted_value):.4f}")

# Show feature values
print("\nFeature values (scaled):")
sample_features = pd.DataFrame({
    'Feature': feature_columns,
    'Scaled_Value': sample[0],
    'Original_Value': scaler.inverse_transform(sample)[0]
})
print(sample_features.to_string(index=False))

# Calculate contribution using feature importance
feature_contributions = sample[0] * rf.feature_importances_
contribution_df = pd.DataFrame({
    'Feature': feature_columns,
    'Contribution': feature_contributions
}).sort_values('Contribution', key=abs, ascending=False)

print("\nFeature Contributions (approximate):")
print(contribution_df.to_string(index=False))

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Feature values
axes[0].barh(sample_features['Feature'], sample_features['Scaled_Value'],
             color='steelblue', edgecolor='black', alpha=0.7)
axes[0].axvline(x=0, color='black', linewidth=1)
axes[0].set_xlabel('Scaled Feature Value', fontsize=12)
axes[0].set_title(f'Sample {sample_idx} Feature Values', fontsize=13, fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='x')

# Contributions
colors = ['green' if x > 0 else 'red' for x in contribution_df['Contribution']]
axes[1].barh(contribution_df['Feature'], contribution_df['Contribution'],
             color=colors, edgecolor='black', alpha=0.7)
axes[1].axvline(x=0, color='black', linewidth=1)
axes[1].set_xlabel('Contribution to Prediction', fontsize=12)
axes[1].set_title('Feature Contributions (Approximate)', fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('individual_prediction.png', dpi=150, bbox_inches='tight')
print("\nIndividual prediction plot saved to 'individual_prediction.png'")

# ============================================================================
# 6. COMPARISON OF INTERPRETATION METHODS
# ============================================================================
print("\n" + "="*70)
print("6. COMPARING INTERPRETATION METHODS")
print("="*70)

# Combine all importance measures
comparison_df = pd.DataFrame({
    'Feature': feature_columns,
    'Built-in (RF)': rf.feature_importances_,
    'Permutation (RF)': perm_importance_rf.importances_mean,
    'Built-in (GB)': gbr.feature_importances_,
    'Permutation (GB)': perm_importance_gbr.importances_mean
})

# Normalize to make comparable
for col in ['Built-in (RF)', 'Permutation (RF)', 'Built-in (GB)', 'Permutation (GB)']:
    comparison_df[f'{col}_norm'] = (comparison_df[col] - comparison_df[col].min()) / \
                                    (comparison_df[col].max() - comparison_df[col].min())

# Rank features
for col in ['Built-in (RF)', 'Permutation (RF)']:
    comparison_df[f'{col}_rank'] = comparison_df[col].rank(ascending=False)

print("\nFeature Importance Comparison:")
print(comparison_df[['Feature', 'Built-in (RF)', 'Permutation (RF)',
                      'Built-in (GB)', 'Permutation (GB)']].to_string(index=False))

# Visualize comparison
fig, ax = plt.subplots(figsize=(12, 8))

x = np.arange(len(feature_columns))
width = 0.2

ax.bar(x - 1.5*width, comparison_df['Built-in (RF)_norm'],
       width, label='Built-in (RF)', alpha=0.8, edgecolor='black')
ax.bar(x - 0.5*width, comparison_df['Permutation (RF)_norm'],
       width, label='Permutation (RF)', alpha=0.8, edgecolor='black')
ax.bar(x + 0.5*width, comparison_df['Built-in (GB)_norm'],
       width, label='Built-in (GB)', alpha=0.8, edgecolor='black')
ax.bar(x + 1.5*width, comparison_df['Permutation (GB)_norm'],
       width, label='Permutation (GB)', alpha=0.8, edgecolor='black')

ax.set_xlabel('Feature', fontsize=12)
ax.set_ylabel('Normalized Importance', fontsize=12)
ax.set_title('Comparison of Feature Importance Methods', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(feature_columns, rotation=45, ha='right')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('importance_methods_comparison.png', dpi=150, bbox_inches='tight')
print("\nImportance methods comparison saved to 'importance_methods_comparison.png'")

# ============================================================================
# 7. BEST PRACTICES & RECOMMENDATIONS
# ============================================================================
print("\n" + "="*70)
print("7. MODEL INTERPRETATION BEST PRACTICES")
print("="*70)

print("""
INTERPRETATION METHODS COMPARISON:

1. BUILT-IN FEATURE IMPORTANCE (Tree-based):
   [OK] Fast, no additional computation
   [OK] Shows importance for training data
   [X] Model-specific (only tree-based)
   [X] Can be biased toward high-cardinality features
   -> Use for: Quick insights, tree-based models

2. PERMUTATION IMPORTANCE:
   [OK] Model-agnostic (works with any model)
   [OK] Uses test data (more realistic)
   [OK] Captures complex relationships
   [X] Computationally expensive
   [X] Can be unstable with correlated features
   -> Use for: Reliable importance ranking, any model

3. PARTIAL DEPENDENCE PLOTS:
   [OK] Shows direction of relationship (increasing/decreasing)
   [OK] Visualizes non-linear effects
   [OK] Intuitive interpretation
   [X] Assumes feature independence (may be misleading)
   [X] Averages over all other features
   -> Use for: Understanding feature effects, presentations

4. TWO-WAY PDP:
   [OK] Reveals feature interactions
   [OK] Shows combined effects
   [X] Computationally expensive
   [X] Difficult to interpret with many features
   -> Use for: Investigating suspected interactions

5. SHAP VALUES (if available):
   [OK] Theoretically sound (game theory)
   [OK] Individual prediction explanations
   [OK] Shows positive/negative contributions
   [X] Very computationally expensive
   [X] Requires additional library
   -> Use for: High-stakes decisions, detailed analysis

PRACTICAL WORKFLOW:

1. Start with built-in importance for quick insights
2. Validate with permutation importance
3. Use PDP to understand relationships
4. Check for interactions with 2D PDP
5. Explain individual predictions when needed

CHEMISTRY-SPECIFIC INSIGHTS:

From our analysis:
- LogP (lipophilicity) is typically most important
- TPSA (polar surface area) affects solubility significantly
- Molecular weight has non-linear effects
- H-bond donors/acceptors show threshold behavior

COMMON PITFALLS TO AVOID:

[X] Don't trust only one interpretation method
[X] Don't ignore feature correlations
[X] Don't over-interpret small importance differences
[X] Don't forget to check on test data
[X] Don't assume causation from importance

WHEN TO USE EACH APPROACH:

Quick exploratory analysis:
  -> Built-in importance + PDP for top features

Model debugging:
  -> Permutation importance + individual predictions

Stakeholder presentation:
  -> PDP plots with clear interpretations

Scientific publication:
  -> Multiple methods + cross-validation

Regulatory approval:
  -> SHAP or permutation importance

TIPS FOR CHEMISTRY:

1. Compare importance with chemical intuition
2. Check if important features align with known mechanisms
3. Investigate unexpected importance rankings
4. Use domain knowledge to validate patterns
5. Consider physical constraints in interpretations

EXERCISE FOR YOU:
1. Install and try SHAP values: pip install shap
2. Create PDPs for all features, not just top ones
3. Analyze predictions for outliers
4. Compare importance across different models
5. Investigate feature interactions for your domain
""")

print("\n" + "="*70)
print("Activity 4 Complete!")
print("="*70)
