"""
Activity 6: Regularization Techniques

DOST-ITDI AI Training Workshop
Learning Objectives:
1. Understand bias-variance tradeoff
2. Apply L1 (Lasso) and L2 (Ridge) regularization
3. Use ElasticNet (combined L1+L2)
4. Tune regularization strength
5. Compare regularization effects on different models
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, learning_curve, validation_curve
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet,
    LogisticRegression, RidgeClassifier
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("Activity 6: Regularization Techniques")
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
df_clean = df.loc[descriptors_df.index].copy()

feature_columns = ['LogP', 'TPSA', 'MolWeight', 'NumHDonors', 'NumHAcceptors']
for col in descriptors_df.columns:
    if col not in df_clean.columns:
        df_clean[col] = descriptors_df[col].values

X = df_clean[feature_columns].values
y = df_clean['measured log solubility in mols per litre'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nDataset: {X.shape}")

# ============================================================================
# 1. BIAS-VARIANCE TRADEOFF DEMONSTRATION
# ============================================================================
print("\n" + "="*70)
print("1. BIAS-VARIANCE TRADEOFF")
print("="*70)

print("\nUnderfitting (High Bias):")
print("  - Model too simple")
print("  - Poor performance on both train and test")
print("  - Solution: Add complexity or reduce regularization")

print("\nOverfitting (High Variance):")
print("  - Model too complex")
print("  - Great on train, poor on test")
print("  - Solution: Add regularization or reduce complexity")

print("\nOptimal:")
print("  - Balanced complexity")
print("  - Good on both train and test")

# Demonstrate with polynomial features
X_simple = X[:, :2]  # Use only 2 features
X_train_simple, X_test_simple, y_train_simple, y_test_simple = train_test_split(
    X_simple, y, test_size=0.2, random_state=42
)

scaler_simple = StandardScaler()
X_train_simple_scaled = scaler_simple.fit_transform(X_train_simple)
X_test_simple_scaled = scaler_simple.transform(X_test_simple)

results_complexity = []

for degree in [1, 2, 3, 5, 10]:
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_poly = poly.fit_transform(X_train_simple_scaled)
    X_test_poly = poly.transform(X_test_simple_scaled)

    model = LinearRegression()
    model.fit(X_train_poly, y_train_simple)

    train_r2 = r2_score(y_train_simple, model.predict(X_train_poly))
    test_r2 = r2_score(y_test_simple, model.predict(X_test_poly))

    results_complexity.append({
        'Degree': degree,
        'N_Features': X_train_poly.shape[1],
        'Train_R2': train_r2,
        'Test_R2': test_r2,
        'Gap': train_r2 - test_r2
    })

    status = "Underfit" if test_r2 < 0.7 else ("Overfit" if train_r2 - test_r2 > 0.1 else "Good")
    print(f"\nPolynomial Degree {degree} ({X_train_poly.shape[1]} features):")
    print(f"  Train R²: {train_r2:.4f}")
    print(f"  Test R²: {test_r2:.4f}")
    print(f"  Gap: {train_r2 - test_r2:.4f} → {status}")

complexity_df = pd.DataFrame(results_complexity)

# Visualize bias-variance tradeoff
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(complexity_df['Degree'], complexity_df['Train_R2'],
             'o-', linewidth=2, markersize=10, label='Train R²', color='blue')
axes[0].plot(complexity_df['Degree'], complexity_df['Test_R2'],
             's-', linewidth=2, markersize=10, label='Test R²', color='red')
axes[0].set_xlabel('Polynomial Degree (Complexity)', fontsize=12)
axes[0].set_ylabel('R² Score', fontsize=12)
axes[0].set_title('Bias-Variance Tradeoff', fontsize=13, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].axvspan(0.5, 2.5, alpha=0.2, color='red', label='Underfit')
axes[0].axvspan(4.5, 10.5, alpha=0.2, color='orange', label='Overfit')

axes[1].bar(complexity_df['Degree'], complexity_df['Gap'],
            color='coral', edgecolor='black', alpha=0.7)
axes[1].set_xlabel('Polynomial Degree', fontsize=12)
axes[1].set_ylabel('Train-Test Gap', fontsize=12)
axes[1].set_title('Overfitting Indicator (Gap)', fontsize=13, fontweight='bold')
axes[1].axhline(y=0.1, color='red', linestyle='--', label='Overfitting Threshold')
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('bias_variance_tradeoff.png', dpi=150, bbox_inches='tight')
print("\nBias-variance tradeoff plot saved to 'bias_variance_tradeoff.png'")

# ============================================================================
# 2. L2 REGULARIZATION (RIDGE)
# ============================================================================
print("\n" + "="*70)
print("2. L2 REGULARIZATION (RIDGE) - Shrinking Coefficients")
print("="*70)

print("\nRidge penalty: α × Σ(coefficient²)")
print("  - Shrinks all coefficients toward zero")
print("  - Keeps all features (no feature selection)")
print("  - Good when many features are somewhat useful")

# Try different alpha values
alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
ridge_results = []

for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train_scaled, y_train)

    train_r2 = r2_score(y_train, ridge.predict(X_train_scaled))
    test_r2 = r2_score(y_test, ridge.predict(X_test_scaled))
    coef_norm = np.linalg.norm(ridge.coef_)  # L2 norm of coefficients

    ridge_results.append({
        'Alpha': alpha,
        'Train_R2': train_r2,
        'Test_R2': test_r2,
        'Coef_Norm': coef_norm
    })

    print(f"\nα={alpha:7.3f}: Train R²={train_r2:.4f}, Test R²={test_r2:.4f}, ||coef||={coef_norm:.4f}")

ridge_df = pd.DataFrame(ridge_results)

# Find optimal alpha
best_idx = ridge_df['Test_R2'].idxmax()
best_alpha = ridge_df.loc[best_idx, 'Alpha']
print(f"\nBest α: {best_alpha} (Test R²={ridge_df.loc[best_idx, 'Test_R2']:.4f})")

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].semilogx(ridge_df['Alpha'], ridge_df['Train_R2'],
                 'o-', linewidth=2, label='Train R²', color='blue')
axes[0].semilogx(ridge_df['Alpha'], ridge_df['Test_R2'],
                 's-', linewidth=2, label='Test R²', color='red')
axes[0].axvline(x=best_alpha, color='green', linestyle='--', label=f'Best α={best_alpha}')
axes[0].set_xlabel('Regularization Strength (α)', fontsize=12)
axes[0].set_ylabel('R² Score', fontsize=12)
axes[0].set_title('Ridge: Effect of Regularization', fontsize=13, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].semilogx(ridge_df['Alpha'], ridge_df['Coef_Norm'],
                 'o-', linewidth=2, color='purple')
axes[1].set_xlabel('Regularization Strength (α)', fontsize=12)
axes[1].set_ylabel('Coefficient Norm ||coef||', fontsize=12)
axes[1].set_title('Ridge: Coefficient Shrinkage', fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ridge_regularization.png', dpi=150, bbox_inches='tight')
print("\nRidge regularization plot saved to 'ridge_regularization.png'")

# ============================================================================
# 3. L1 REGULARIZATION (LASSO) - Feature Selection
# ============================================================================
print("\n" + "="*70)
print("3. L1 REGULARIZATION (LASSO) - Automatic Feature Selection")
print("="*70)

print("\nLasso penalty: α × Σ|coefficient|")
print("  - Forces some coefficients to exactly zero")
print("  - Performs automatic feature selection")
print("  - Good when many features are irrelevant")

lasso_results = []
lasso_models = {}

for alpha in alphas:
    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(X_train_scaled, y_train)

    train_r2 = r2_score(y_train, lasso.predict(X_train_scaled))
    test_r2 = r2_score(y_test, lasso.predict(X_test_scaled))
    n_nonzero = np.sum(lasso.coef_ != 0)

    lasso_results.append({
        'Alpha': alpha,
        'Train_R2': train_r2,
        'Test_R2': test_r2,
        'N_Features': n_nonzero
    })

    lasso_models[alpha] = lasso

    print(f"\nα={alpha:7.3f}: Train R²={train_r2:.4f}, Test R²={test_r2:.4f}, Active features={n_nonzero}/{len(feature_columns)}")

lasso_df = pd.DataFrame(lasso_results)

# Visualize coefficient paths
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Coefficient paths
for i, feature in enumerate(feature_columns):
    coefs = [lasso_models[alpha].coef_[i] for alpha in alphas]
    axes[0].semilogx(alphas, coefs, 'o-', linewidth=2, label=feature)

axes[0].set_xlabel('Regularization Strength (α)', fontsize=12)
axes[0].set_ylabel('Coefficient Value', fontsize=12)
axes[0].set_title('Lasso: Coefficient Paths', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)
axes[0].axhline(y=0, color='black', linewidth=1)

# Number of features vs alpha
axes[1].semilogx(lasso_df['Alpha'], lasso_df['N_Features'],
                 'o-', linewidth=2, color='steelblue')
axes[1].set_xlabel('Regularization Strength (α)', fontsize=12)
axes[1].set_ylabel('Number of Active Features', fontsize=12)
axes[1].set_title('Lasso: Feature Selection', fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lasso_regularization.png', dpi=150, bbox_inches='tight')
print("\nLasso regularization plot saved to 'lasso_regularization.png'")

# ============================================================================
# 4. ELASTICNET - COMBINING L1 AND L2
# ============================================================================
print("\n" + "="*70)
print("4. ELASTICNET - Best of Both Worlds")
print("="*70)

print("\nElasticNet penalty: α × [λ × Σ|coef| + (1-λ) × Σ(coef²)]")
print("  - Combines L1 (feature selection) and L2 (shrinkage)")
print("  - l1_ratio controls balance (0=Ridge, 1=Lasso)")
print("  - Good when features are correlated")

# Try different l1_ratios
l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
alpha = 1.0

elastic_results = []

for l1_ratio in l1_ratios:
    elastic = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000)
    elastic.fit(X_train_scaled, y_train)

    train_r2 = r2_score(y_train, elastic.predict(X_train_scaled))
    test_r2 = r2_score(y_test, elastic.predict(X_test_scaled))
    n_nonzero = np.sum(elastic.coef_ != 0)

    elastic_results.append({
        'L1_Ratio': l1_ratio,
        'Train_R2': train_r2,
        'Test_R2': test_r2,
        'N_Features': n_nonzero
    })

    l1_pct = l1_ratio * 100
    l2_pct = (1 - l1_ratio) * 100
    print(f"\nL1={l1_pct:.0f}%, L2={l2_pct:.0f}%: Train R²={train_r2:.4f}, Test R²={test_r2:.4f}, Features={n_nonzero}")

elastic_df = pd.DataFrame(elastic_results)

plt.figure(figsize=(10, 6))
plt.plot(elastic_df['L1_Ratio'], elastic_df['Test_R2'],
         'o-', linewidth=2, markersize=10, color='purple')
plt.xlabel('L1 Ratio (0=Ridge, 1=Lasso)', fontsize=12)
plt.ylabel('Test R² Score', fontsize=12)
plt.title('ElasticNet: Effect of L1/L2 Mix', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('elasticnet_l1_ratio.png', dpi=150, bbox_inches='tight')
print("\nElasticNet plot saved to 'elasticnet_l1_ratio.png'")

# ============================================================================
# 5. COMPARISON OF ALL REGULARIZATION METHODS
# ============================================================================
print("\n" + "="*70)
print("5. COMPARISON: LINEAR vs RIDGE vs LASSO vs ELASTICNET")
print("="*70)

# Train all models with optimal settings
models = {
    'Linear': LinearRegression(),
    'Ridge': Ridge(alpha=best_alpha),
    'Lasso': Lasso(alpha=0.01, max_iter=10000),
    'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000)
}

comparison_results = []

for name, model in models.items():
    model.fit(X_train_scaled, y_train)

    train_r2 = r2_score(y_train, model.predict(X_train_scaled))
    test_r2 = r2_score(y_test, model.predict(X_test_scaled))

    if hasattr(model, 'coef_'):
        coef = model.coef_
        n_nonzero = np.sum(coef != 0) if name != 'Linear' else len(coef)
        max_coef = np.max(np.abs(coef))
    else:
        n_nonzero = len(feature_columns)
        max_coef = 0

    comparison_results.append({
        'Model': name,
        'Train_R2': train_r2,
        'Test_R2': test_r2,
        'Gap': train_r2 - test_r2,
        'Active_Features': n_nonzero,
        'Max_|Coef|': max_coef
    })

comparison_df = pd.DataFrame(comparison_results)

print("\nRegularization Methods Comparison:")
print(comparison_df.to_string(index=False))

# Visualize comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()

# R² scores
x = np.arange(len(comparison_df))
width = 0.35
axes[0].bar(x - width/2, comparison_df['Train_R2'], width, label='Train', alpha=0.8)
axes[0].bar(x + width/2, comparison_df['Test_R2'], width, label='Test', alpha=0.8)
axes[0].set_xlabel('Model', fontsize=11)
axes[0].set_ylabel('R² Score', fontsize=11)
axes[0].set_title('R² Score Comparison', fontsize=12, fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(comparison_df['Model'])
axes[0].legend()
axes[0].grid(True, alpha=0.3, axis='y')

# Train-Test Gap
axes[1].bar(comparison_df['Model'], comparison_df['Gap'],
            color='coral', edgecolor='black', alpha=0.7)
axes[1].set_xlabel('Model', fontsize=11)
axes[1].set_ylabel('Train-Test Gap', fontsize=11)
axes[1].set_title('Overfitting Indicator', fontsize=12, fontweight='bold')
axes[1].axhline(y=0.05, color='red', linestyle='--', label='Acceptable')
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis='y')

# Active features
axes[2].bar(comparison_df['Model'], comparison_df['Active_Features'],
            color='steelblue', edgecolor='black', alpha=0.7)
axes[2].set_xlabel('Model', fontsize=11)
axes[2].set_ylabel('Number of Features', fontsize=11)
axes[2].set_title('Feature Selection', fontsize=12, fontweight='bold')
axes[2].axhline(y=len(feature_columns), color='red', linestyle='--', label='Total Features')
axes[2].legend()
axes[2].grid(True, alpha=0.3, axis='y')

# Coefficient magnitudes
for name, model in models.items():
    if hasattr(model, 'coef_'):
        coef_sorted = np.sort(np.abs(model.coef_))[::-1]
        axes[3].plot(coef_sorted, 'o-', linewidth=2, label=name, alpha=0.7)

axes[3].set_xlabel('Feature Index (sorted)', fontsize=11)
axes[3].set_ylabel('|Coefficient|', fontsize=11)
axes[3].set_title('Coefficient Magnitudes', fontsize=12, fontweight='bold')
axes[3].legend()
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('regularization_comparison.png', dpi=150, bbox_inches='tight')
print("\nRegularization comparison plot saved to 'regularization_comparison.png'")

# ============================================================================
# 6. BEST PRACTICES & RECOMMENDATIONS
# ============================================================================
print("\n" + "="*70)
print("6. REGULARIZATION BEST PRACTICES")
print("="*70)

print("""
REGULARIZATION METHODS COMPARISON:

1. NO REGULARIZATION (Linear Regression):
   ✅ Unbiased estimates
   ✅ Fast, simple
   ❌ Can overfit with many features
   ❌ Unstable with correlated features
   → Use for: Small datasets, few features

2. L2 REGULARIZATION (Ridge):
   ✅ Handles multicollinearity well
   ✅ Keeps all features
   ✅ Stable, smooth solutions
   ❌ Doesn't perform feature selection
   → Use for: Many correlated features, need all features

3. L1 REGULARIZATION (Lasso):
   ✅ Automatic feature selection
   ✅ Interpretable (sparse solutions)
   ✅ Good with irrelevant features
   ❌ Arbitrary choice among correlated features
   ❌ Can be unstable
   → Use for: Feature selection, sparse models

4. ELASTICNET (L1 + L2):
   ✅ Combines benefits of Ridge and Lasso
   ✅ Handles correlated features better than Lasso
   ✅ Feature selection + stability
   ❌ Two hyperparameters to tune
   → Use for: Correlated features + need selection

WHEN TO USE EACH:

Few features (< 50):
  → Start with Linear Regression
  → Add Ridge if overfitting

Many features (> 50):
  → Try Lasso for feature selection
  → Or Ridge if all features matter

Correlated features:
  → Ridge or ElasticNet
  → Avoid pure Lasso

Need interpretability:
  → Lasso (sparse, few features)

Need stability:
  → Ridge (smooth coefficients)

HYPERPARAMETER TUNING:

1. Use cross-validation:
   - GridSearchCV or RandomizedSearchCV
   - Try logarithmic range: [0.001, 0.01, 0.1, 1, 10, 100]

2. Start broad, then refine:
   - First: Wide range to find ballpark
   - Then: Narrow range around best value

3. Monitor train vs test:
   - Too small α → overfitting (high train, low test)
   - Too large α → underfitting (low train and test)

PRACTICAL WORKFLOW:

1. Start with no regularization (baseline)
2. Try Ridge (simple, often works well)
3. If need feature selection → try Lasso
4. If correlated features → try ElasticNet
5. Always cross-validate to choose α

COMMON PITFALLS:

❌ Not scaling features before regularization
❌ Using same α for different data scales
❌ Choosing α on training set (use CV!)
❌ Forgetting that regularization needs tuning
❌ Applying to already regularized models (e.g., Random Forest)

CHEMISTRY-SPECIFIC INSIGHTS:

For molecular descriptors:
  - Often have multicollinearity (LogP, MolWeight, etc.)
  - Ridge or ElasticNet usually work best
  - Lasso good for identifying key descriptors
  - α typically in range [0.01, 10]

For high-dimensional data (many descriptors):
  - Lasso for automatic feature selection
  - Can reduce from 100s to 10s of features
  - Improves interpretability for chemists

REGULARIZATION IN DEEP LEARNING:

Neural networks use different regularization:
  - Dropout: Randomly drop units during training
  - Batch Normalization: Normalize layer inputs
  - Early Stopping: Stop before overfitting
  - L2 Weight Decay: Similar to Ridge

EXERCISE FOR YOU:
1. Try validation_curve() to visualize regularization effect
2. Implement Ridge in PyTorch with weight_decay
3. Compare regularization across different datasets
4. Use RidgeCV or LassoCV for automatic α selection
5. Analyze which features are selected by Lasso
""")

print("\n" + "="*70)
print("Activity 6 Complete!")
print("="*70)
