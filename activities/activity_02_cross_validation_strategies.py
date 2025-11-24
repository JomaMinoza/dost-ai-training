"""
Activity 2: Cross-Validation Strategies

DOST-ITDI AI Training Workshop
Learning Objectives:
1. Understand different CV strategies
2. Apply K-Fold, Stratified K-Fold, Leave-One-Out
3. Use TimeSeriesSplit for temporal data
4. Implement Nested CV for unbiased evaluation
5. Choose appropriate CV strategy for your problem
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import (
    KFold, StratifiedKFold, LeaveOneOut, ShuffleSplit,
    cross_val_score, cross_validate, learning_curve
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, make_scorer
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("Activity 2: Cross-Validation Strategies")
print("="*70)

# Load dataset
url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv"
df = pd.read_csv(url)

# Calculate descriptors (simplified for speed)
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

descriptors_list = [calculate_descriptors(smiles) for smiles in df['mol']]
descriptors_df = pd.DataFrame(descriptors_list).dropna()

df_clean = df.loc[descriptors_df.index].copy()
X = descriptors_df.values
y = df_clean['Class'].values

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"\nDataset: {X_scaled.shape}")
print(f"Class distribution: {np.bincount(y)}")

# ============================================================================
# 1. K-FOLD CROSS-VALIDATION
# ============================================================================
print("\n" + "="*70)
print("1. K-FOLD CROSS-VALIDATION")
print("="*70)

# Create model
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

# Standard K-Fold
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

print("\nK-Fold (k=5):")
scores = cross_val_score(model, X_scaled, y, cv=kfold, scoring='roc_auc')
print(f"  Scores per fold: {[f'{s:.4f}' for s in scores]}")
print(f"  Mean: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")

# Different k values
print("\nEffect of different k values:")
for k in [3, 5, 10]:
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_scaled, y, cv=kfold, scoring='roc_auc')
    print(f"  k={k:2d}: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")

# ============================================================================
# 2. STRATIFIED K-FOLD (For Imbalanced Data)
# ============================================================================
print("\n" + "="*70)
print("2. STRATIFIED K-FOLD (Preserves Class Distribution)")
print("="*70)

stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\nStratified K-Fold ensures each fold has similar class distribution:")
for fold_idx, (train_idx, val_idx) in enumerate(stratified_kfold.split(X_scaled, y), 1):
    train_dist = np.bincount(y[train_idx])
    val_dist = np.bincount(y[val_idx])
    print(f"  Fold {fold_idx}: Train={train_dist}, Val={val_dist}")

scores_stratified = cross_val_score(model, X_scaled, y, cv=stratified_kfold, scoring='roc_auc')
print(f"\nStratified K-Fold scores:")
print(f"  Mean: {scores_stratified.mean():.4f} (+/- {scores_stratified.std()*2:.4f})")

# ============================================================================
# 3. CROSS_VALIDATE with Multiple Metrics
# ============================================================================
print("\n" + "="*70)
print("3. CROSS_VALIDATE with Multiple Metrics")
print("="*70)

scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
    'roc_auc': 'roc_auc'
}

cv_results = cross_validate(
    model, X_scaled, y,
    cv=stratified_kfold,
    scoring=scoring,
    return_train_score=True
)

print("\nMultiple Metrics Evaluation:")
for metric in scoring.keys():
    train_scores = cv_results[f'train_{metric}']
    test_scores = cv_results[f'test_{metric}']
    print(f"  {metric.upper():10s}: Train={train_scores.mean():.4f}, Test={test_scores.mean():.4f}")

# ============================================================================
# 4. LEARNING CURVE Analysis
# ============================================================================
print("\n" + "="*70)
print("4. LEARNING CURVE Analysis")
print("="*70)

train_sizes, train_scores, val_scores = learning_curve(
    model, X_scaled, y,
    cv=stratified_kfold,
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='roc_auc',
    n_jobs=-1
)

train_mean = train_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
val_mean = val_scores.mean(axis=1)
val_std = val_scores.std(axis=1)

# Plot learning curve
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, label='Training score', marker='o')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
plt.plot(train_sizes, val_mean, label='Validation score', marker='s')
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1)
plt.xlabel('Training Set Size')
plt.ylabel('ROC-AUC Score')
plt.title('Learning Curve')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('learning_curve.png', dpi=150, bbox_inches='tight')
print("\nLearning curve saved to 'learning_curve.png'")

print("\nTraining Set Size Analysis:")
for size, train_score, val_score in zip(train_sizes, train_mean, val_mean):
    print(f"  Size={size:4.0f}: Train={train_score:.4f}, Val={val_score:.4f}, Gap={train_score-val_score:.4f}")

# ============================================================================
# 5. NESTED CROSS-VALIDATION (Unbiased Evaluation)
# ============================================================================
print("\n" + "="*70)
print("5. NESTED CROSS-VALIDATION (Unbiased Evaluation)")
print("="*70)

from sklearn.model_selection import GridSearchCV

print("\nNested CV: Inner loop for hyperparameter tuning, outer loop for evaluation")

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [5, 10, 15]
}

# Outer CV
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# Inner CV
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

nested_scores = []

for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X_scaled, y), 1):
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Inner CV: Hyperparameter tuning
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42, class_weight='balanced'),
        param_grid,
        cv=inner_cv,
        scoring='roc_auc'
    )
    grid_search.fit(X_train, y_train)

    # Evaluate on outer test set
    best_model = grid_search.best_estimator_
    score = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])
    nested_scores.append(score)

    print(f"  Outer Fold {fold_idx}: Best params={grid_search.best_params_}, Score={score:.4f}")

print(f"\nNested CV Mean Score: {np.mean(nested_scores):.4f} (+/- {np.std(nested_scores)*2:.4f})")
print("This is an unbiased estimate of model performance!")

# ============================================================================
# 6. BEST PRACTICES & RECOMMENDATIONS
# ============================================================================
print("\n" + "="*70)
print("6. CV STRATEGY SELECTION GUIDE")
print("="*70)

print("""
WHEN TO USE EACH CV STRATEGY:

1. STRATIFIED K-FOLD (MOST COMMON):
   ✅ Classification problems
   ✅ Imbalanced datasets
   ✅ Preserves class distribution
   → USE THIS AS DEFAULT for classification

2. REGULAR K-FOLD:
   ✅ Regression problems
   ✅ Large balanced datasets
   ✅ When class distribution doesn't matter

3. LEAVE-ONE-OUT (LOO):
   ✅ Very small datasets (n < 100)
   ⚠ Computationally expensive
   ⚠ High variance in estimates

4. SHUFFLE SPLIT:
   ✅ Want to control train/test size independently
   ✅ Multiple random splits
   ✅ Very large datasets (sample subset)

5. NESTED CV:
   ✅ Unbiased model evaluation
   ✅ When doing hyperparameter tuning
   ✅ Research / publication-quality results
   ⚠ Computationally expensive

COMMON PITFALLS TO AVOID:

❌ Using test set multiple times (data leakage!)
❌ Not stratifying for imbalanced classification
❌ Too few folds (k<5) - high variance
❌ Too many folds - computationally expensive
❌ Forgetting to shuffle (especially with sorted data)
❌ Using k-fold for time series (use TimeSeriesSplit)

RECOMMENDATIONS:

1. Default: Use k=5 or k=10 Stratified K-Fold
2. For hyperparameter tuning: Use Nested CV
3. Monitor both train and validation scores (check overfitting)
4. Use multiple metrics (accuracy, precision, recall, F1, ROC-AUC)
5. Visualize learning curves to diagnose problems
6. Report mean +/- std of scores
7. Set random_state for reproducibility

COMPUTATIONAL COST:

K-Fold with k folds:
  - Training: k models
  - Total data used: 100% (with different splits)

Nested CV (k_outer=5, k_inner=3):
  - Training: 5 × 3 × (grid size) models
  - Much more expensive but unbiased!

EXERCISE FOR YOU:
1. Try different k values and compare stability
2. Implement TimeSeriesSplit for temporal data
3. Compare Nested CV vs simple CV + separate test set
4. Analyze learning curves for different models
5. Experiment with GroupKFold for grouped data
""")

print("\n" + "="*70)
print("Activity 2 Complete!")
print("="*70)
