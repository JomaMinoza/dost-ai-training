"""
Activity 1: Hyperparameter Tuning with Grid Search and Random Search

DOST-ITDI AI Training Workshop
Learning Objectives:
1. Understand hyperparameter tuning
2. Use GridSearchCV for exhaustive search
3. Use RandomizedSearchCV for efficient search
4. Compare different search strategies
5. Apply best practices for tuning
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("Activity 1: Hyperparameter Tuning")
print("="*70)

# Load BACE dataset
url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv"
df = pd.read_csv(url)

print(f"\nDataset loaded: {df.shape}")
print(f"Class distribution:\n{df['Class'].value_counts()}")

# Calculate molecular descriptors
from rdkit import Chem
from rdkit.Chem import Descriptors

def calculate_descriptors(smiles):
    """Calculate molecular descriptors"""
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
            'FractionCSP3': Descriptors.FractionCSP3(mol)
        }
    except:
        return None

print("\nCalculating molecular descriptors...")
descriptors_list = [calculate_descriptors(smiles) for smiles in df['mol']]
descriptors_df = pd.DataFrame(descriptors_list).dropna()

# Prepare data
df_clean = df.loc[descriptors_df.index].copy()
X = descriptors_df.values
y = df_clean['Class'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTraining set: {X_train_scaled.shape}")
print(f"Test set: {X_test_scaled.shape}")

# ============================================================================
# 1. GRID SEARCH CV - Exhaustive Search
# ============================================================================
print("\n" + "="*70)
print("1. GRID SEARCH CV - Exhaustive Search")
print("="*70)

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'class_weight': ['balanced', None]
}

print(f"\nParameter grid:")
for param, values in param_grid.items():
    print(f"  {param}: {values}")

total_combinations = np.prod([len(v) for v in param_grid.values()])
print(f"\nTotal combinations to test: {total_combinations}")
print(f"With 5-fold CV: {total_combinations * 5} fits")

# Create GridSearchCV
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

print("\nRunning Grid Search (this may take a while)...")
grid_search.fit(X_train_scaled, y_train)

print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best CV score (ROC-AUC): {grid_search.best_score_:.4f}")

# Evaluate on test set
y_pred_grid = grid_search.predict(X_test_scaled)
y_prob_grid = grid_search.predict_proba(X_test_scaled)[:, 1]
test_score_grid = roc_auc_score(y_test, y_prob_grid)

print(f"\nTest set performance:")
print(f"  ROC-AUC: {test_score_grid:.4f}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred_grid, target_names=['Inactive', 'Active']))

# ============================================================================
# 2. RANDOM SEARCH CV - Efficient Search
# ============================================================================
print("\n" + "="*70)
print("2. RANDOM SEARCH CV - Efficient Search")
print("="*70)

# Define parameter distributions for random search
param_distributions = {
    'n_estimators': [50, 100, 150, 200, 250, 300],
    'max_depth': [3, 5, 7, 10, 15, 20, None],
    'min_samples_split': np.arange(2, 21),
    'min_samples_leaf': np.arange(1, 11),
    'max_features': ['sqrt', 'log2', None, 0.5, 0.7],
    'class_weight': ['balanced', 'balanced_subsample', None]
}

print(f"\nParameter distributions:")
for param, values in param_distributions.items():
    if isinstance(values, list):
        print(f"  {param}: {values}")
    else:
        print(f"  {param}: {values.min()}-{values.max()}")

# Create RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_distributions,
    n_iter=50,  # Number of parameter settings sampled
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1,
    random_state=42
)

print(f"\nSampling {50} random combinations (much faster than {total_combinations})...")
print("Running Random Search...")
random_search.fit(X_train_scaled, y_train)

print(f"\nBest parameters: {random_search.best_params_}")
print(f"Best CV score (ROC-AUC): {random_search.best_score_:.4f}")

# Evaluate on test set
y_pred_random = random_search.predict(X_test_scaled)
y_prob_random = random_search.predict_proba(X_test_scaled)[:, 1]
test_score_random = roc_auc_score(y_test, y_prob_random)

print(f"\nTest set performance:")
print(f"  ROC-AUC: {test_score_random:.4f}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred_random, target_names=['Inactive', 'Active']))

# ============================================================================
# 3. COMPARISON & ANALYSIS
# ============================================================================
print("\n" + "="*70)
print("3. COMPARISON & ANALYSIS")
print("="*70)

# Compare results
comparison_df = pd.DataFrame({
    'Method': ['Grid Search', 'Random Search'],
    'Best CV Score': [grid_search.best_score_, random_search.best_score_],
    'Test Score': [test_score_grid, test_score_random],
    'Combinations Tested': [total_combinations * 5, 50 * 5]
})

print("\nPerformance Comparison:")
print(comparison_df.to_string(index=False))

# Show top 10 parameter combinations from grid search
print("\nTop 10 Parameter Combinations (Grid Search):")
results_df = pd.DataFrame(grid_search.cv_results_)
top_10 = results_df.nlargest(10, 'mean_test_score')[
    ['params', 'mean_test_score', 'std_test_score', 'rank_test_score']
]
for idx, row in top_10.iterrows():
    print(f"\n{row['rank_test_score']}. Score: {row['mean_test_score']:.4f} (+/- {row['std_test_score']:.4f})")
    print(f"   Parameters: {row['params']}")

# ============================================================================
# 4. BEST PRACTICES & RECOMMENDATIONS
# ============================================================================
print("\n" + "="*70)
print("4. BEST PRACTICES & RECOMMENDATIONS")
print("="*70)

print("""
KEY TAKEAWAYS:

1. WHEN TO USE GRID SEARCH:
   - Small parameter space
   - You have computational resources
   - Need exhaustive search
   - Critical application (requires best possible model)

2. WHEN TO USE RANDOM SEARCH:
   - Large parameter space
   - Limited time/resources
   - Exploring wide range of values
   - Often finds good solutions faster

3. TIPS FOR HYPERPARAMETER TUNING:
   - Start with Random Search for exploration
   - Use Grid Search around promising regions
   - Monitor cross-validation scores
   - Check for overfitting (CV vs Test score gap)
   - Use appropriate scoring metric for your problem
   - Consider Bayesian Optimization for complex problems

4. COMMON PARAMETERS TO TUNE:
   Random Forest:
   - n_estimators: More trees = better, but diminishing returns
   - max_depth: Controls overfitting
   - min_samples_split/leaf: Prevents overfitting on small samples
   - max_features: Randomness in feature selection
   - class_weight: Handle imbalanced data

5. ADVANCED TECHNIQUES:
   - Bayesian Optimization (Optuna, Hyperopt)
   - Successive Halving (HalvingGridSearchCV)
   - Early stopping for iterative models
   - Nested cross-validation for unbiased evaluation

EXERCISE FOR YOU:
1. Try different scoring metrics ('accuracy', 'f1', 'precision', 'recall')
2. Add more parameters to tune
3. Experiment with different models (GradientBoosting, SVM)
4. Compare computation time vs performance gain
5. Try Bayesian Optimization with Optuna
""")

print("\n" + "="*70)
print("Activity 1 Complete!")
print("="*70)
