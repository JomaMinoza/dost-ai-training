"""
Activity 5: Ensemble Methods

DOST-ITDI AI Training Workshop
Learning Objectives:
1. Understand Voting (averaging predictions)
2. Apply Bagging for variance reduction
3. Use Boosting for bias reduction
4. Implement Stacking (meta-learning)
5. Compare ensemble strategies
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    VotingRegressor, VotingClassifier,
    BaggingRegressor, BaggingClassifier,
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier,
    AdaBoostRegressor, AdaBoostClassifier,
    StackingRegressor, StackingClassifier
)
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVR, SVC
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("Activity 5: Ensemble Methods")
print("="*70)

# ============================================================================
# PART 1: REGRESSION ENSEMBLES
# ============================================================================
print("\n" + "="*70)
print("PART 1: REGRESSION ENSEMBLES (Solubility Prediction)")
print("="*70)

# Load ESOL dataset
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

feature_columns = [
    'Molecular Weight', 'Number of H-Bond Donors', 'Number of Rings',
    'Number of Rotatable Bonds', 'Polar Surface Area'
]
for col in descriptors_df.columns:
    if col not in df_clean.columns:
        df_clean[col] = descriptors_df[col].values

feature_columns.extend(['MolWeight', 'LogP', 'NumHAcceptors', 'TPSA'])

X_reg = df_clean[feature_columns].values
y_reg = df_clean['measured log solubility in mols per litre'].values

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

scaler_reg = StandardScaler()
X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)
X_test_reg_scaled = scaler_reg.transform(X_test_reg)

print(f"\nRegression dataset: {X_reg.shape}")

# ============================================================================
# 1. VOTING REGRESSOR (Simple Averaging)
# ============================================================================
print("\n" + "="*70)
print("1. VOTING REGRESSOR - Averaging Multiple Models")
print("="*70)

# Define base models
lr = LinearRegression()
ridge = Ridge(alpha=1.0)
rf = RandomForestRegressor(n_estimators=50, random_state=42)

# Train individual models
print("\nIndividual Model Performance:")
for name, model in [('Linear', lr), ('Ridge', ridge), ('RF', rf)]:
    model.fit(X_train_reg_scaled, y_train_reg)
    y_pred = model.predict(X_test_reg_scaled)
    r2 = r2_score(y_test_reg, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred))
    print(f"  {name:10s}: R²={r2:.4f}, RMSE={rmse:.4f}")

# Create voting regressor
voting_reg = VotingRegressor([
    ('lr', lr),
    ('ridge', ridge),
    ('rf', rf)
])

voting_reg.fit(X_train_reg_scaled, y_train_reg)
y_pred_voting = voting_reg.predict(X_test_reg_scaled)
r2_voting = r2_score(y_test_reg, y_pred_voting)
rmse_voting = np.sqrt(mean_squared_error(y_test_reg, y_pred_voting))

print(f"\nVoting Regressor (Average):")
print(f"  R²={r2_voting:.4f}, RMSE={rmse_voting:.4f}")
print(f"  → Often better than individual models!")

# ============================================================================
# 2. BAGGING - Bootstrap Aggregating
# ============================================================================
print("\n" + "="*70)
print("2. BAGGING - Reducing Variance via Bootstrap Sampling")
print("="*70)

# Single decision tree (high variance)
dt = DecisionTreeRegressor(max_depth=10, random_state=42)
dt.fit(X_train_reg_scaled, y_train_reg)
y_pred_dt = dt.predict(X_test_reg_scaled)
r2_dt = r2_score(y_test_reg, y_pred_dt)

print(f"\nSingle Decision Tree: R²={r2_dt:.4f} (may overfit)")

# Bagging with multiple trees
bagging_reg = BaggingRegressor(
    estimator=DecisionTreeRegressor(max_depth=10),
    n_estimators=50,
    max_samples=0.8,
    max_features=0.8,
    random_state=42,
    n_jobs=-1
)

bagging_reg.fit(X_train_reg_scaled, y_train_reg)
y_pred_bagging = bagging_reg.predict(X_test_reg_scaled)
r2_bagging = r2_score(y_test_reg, y_pred_bagging)

print(f"Bagging (50 trees): R²={r2_bagging:.4f} (reduced overfitting)")
print(f"  → Improvement: {r2_bagging - r2_dt:+.4f}")

# Random Forest (bagging + feature randomness)
rf_full = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_full.fit(X_train_reg_scaled, y_train_reg)
y_pred_rf = rf_full.predict(X_test_reg_scaled)
r2_rf = r2_score(y_test_reg, y_pred_rf)

print(f"Random Forest: R²={r2_rf:.4f} (bagging + feature randomness)")

# ============================================================================
# 3. BOOSTING - Sequential Error Correction
# ============================================================================
print("\n" + "="*70)
print("3. BOOSTING - Sequentially Reducing Bias")
print("="*70)

# AdaBoost
adaboost_reg = AdaBoostRegressor(
    estimator=DecisionTreeRegressor(max_depth=5),
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
)

adaboost_reg.fit(X_train_reg_scaled, y_train_reg)
y_pred_ada = adaboost_reg.predict(X_test_reg_scaled)
r2_ada = r2_score(y_test_reg, y_pred_ada)

print(f"\nAdaBoost: R²={r2_ada:.4f}")

# Gradient Boosting
gb_reg = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)

gb_reg.fit(X_train_reg_scaled, y_train_reg)
y_pred_gb = gb_reg.predict(X_test_reg_scaled)
r2_gb = r2_score(y_test_reg, y_pred_gb)

print(f"Gradient Boosting: R²={r2_gb:.4f}")
print(f"\n→ Boosting reduces bias by focusing on difficult examples")

# ============================================================================
# 4. STACKING - Meta-Learning
# ============================================================================
print("\n" + "="*70)
print("4. STACKING - Training a Meta-Model")
print("="*70)

# Define base models
base_models = [
    ('rf', RandomForestRegressor(n_estimators=50, random_state=42)),
    ('gb', GradientBoostingRegressor(n_estimators=50, random_state=42)),
    ('ridge', Ridge(alpha=1.0))
]

# Meta-model
meta_model = Ridge(alpha=10.0)

# Create stacking regressor
stacking_reg = StackingRegressor(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5
)

print("\nStacking Process:")
print("  Step 1: Train base models with cross-validation")
print("  Step 2: Use base model predictions as features")
print("  Step 3: Train meta-model on these features")

stacking_reg.fit(X_train_reg_scaled, y_train_reg)
y_pred_stack = stacking_reg.predict(X_test_reg_scaled)
r2_stack = r2_score(y_test_reg, y_pred_stack)

print(f"\nStacking Performance: R²={r2_stack:.4f}")
print(f"  → Combines strengths of multiple models")

# ============================================================================
# 5. COMPARISON OF ALL REGRESSION ENSEMBLES
# ============================================================================
print("\n" + "="*70)
print("5. REGRESSION ENSEMBLE COMPARISON")
print("="*70)

results_reg = pd.DataFrame({
    'Method': [
        'Single Tree', 'Bagging', 'Random Forest',
        'AdaBoost', 'Gradient Boosting', 'Voting', 'Stacking'
    ],
    'R2_Score': [r2_dt, r2_bagging, r2_rf, r2_ada, r2_gb, r2_voting, r2_stack],
    'Type': [
        'Baseline', 'Bagging', 'Bagging',
        'Boosting', 'Boosting', 'Voting', 'Stacking'
    ]
})

results_reg = results_reg.sort_values('R2_Score', ascending=False)
print("\nRegression Ensemble Performance Ranking:")
print(results_reg.to_string(index=False))

# Visualize
plt.figure(figsize=(12, 6))
colors = {'Baseline': 'red', 'Bagging': 'steelblue',
          'Boosting': 'coral', 'Voting': 'green', 'Stacking': 'purple'}
bar_colors = [colors[t] for t in results_reg['Type']]

plt.barh(results_reg['Method'], results_reg['R2_Score'],
         color=bar_colors, edgecolor='black', alpha=0.7)
plt.xlabel('R² Score', fontsize=12)
plt.title('Regression Ensemble Methods Comparison', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='x')

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=colors[t], label=t) for t in colors.keys()]
plt.legend(handles=legend_elements, loc='lower right')

plt.tight_layout()
plt.savefig('ensemble_regression_comparison.png', dpi=150, bbox_inches='tight')
print("\nComparison saved to 'ensemble_regression_comparison.png'")

# ============================================================================
# PART 2: CLASSIFICATION ENSEMBLES
# ============================================================================
print("\n" + "="*70)
print("PART 2: CLASSIFICATION ENSEMBLES (Drug Activity)")
print("="*70)

# Load BACE dataset
url_class = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv"
df_class = pd.read_csv(url_class)

# Prepare features
descriptors_class = [calculate_descriptors(smiles) for smiles in df_class['mol']]
descriptors_df_class = pd.DataFrame(descriptors_class).dropna()
df_class_clean = df_class.loc[descriptors_df_class.index].copy()

for col in descriptors_df_class.columns:
    df_class_clean[col] = descriptors_df_class[col].values

feature_cols_class = ['MolWeight', 'LogP', 'NumHDonors', 'NumHAcceptors', 'TPSA']

X_class = df_class_clean[feature_cols_class].values
y_class = df_class_clean['Class'].values

X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(
    X_class, y_class, test_size=0.2, random_state=42, stratify=y_class
)

scaler_class = StandardScaler()
X_train_class_scaled = scaler_class.fit_transform(X_train_class)
X_test_class_scaled = scaler_class.transform(X_test_class)

print(f"\nClassification dataset: {X_class.shape}")
print(f"Class distribution: {np.bincount(y_class)}")

# Voting Classifier
lr_clf = LogisticRegression(random_state=42, max_iter=1000)
rf_clf = RandomForestClassifier(n_estimators=50, random_state=42)
gb_clf = GradientBoostingClassifier(n_estimators=50, random_state=42)

voting_clf = VotingClassifier([
    ('lr', lr_clf),
    ('rf', rf_clf),
    ('gb', gb_clf)
], voting='soft')  # 'soft' uses probabilities

voting_clf.fit(X_train_class_scaled, y_train_class)
y_pred_voting_clf = voting_clf.predict(X_test_class_scaled)
y_proba_voting_clf = voting_clf.predict_proba(X_test_class_scaled)[:, 1]

acc_voting = accuracy_score(y_test_class, y_pred_voting_clf)
auc_voting = roc_auc_score(y_test_class, y_proba_voting_clf)

print(f"\nVoting Classifier: Accuracy={acc_voting:.4f}, ROC-AUC={auc_voting:.4f}")

# Stacking Classifier
base_clf = [
    ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=50, random_state=42))
]

meta_clf = LogisticRegression()

stacking_clf = StackingClassifier(
    estimators=base_clf,
    final_estimator=meta_clf,
    cv=5
)

stacking_clf.fit(X_train_class_scaled, y_train_class)
y_pred_stack_clf = stacking_clf.predict(X_test_class_scaled)
y_proba_stack_clf = stacking_clf.predict_proba(X_test_class_scaled)[:, 1]

acc_stack = accuracy_score(y_test_class, y_pred_stack_clf)
auc_stack = roc_auc_score(y_test_class, y_proba_stack_clf)

print(f"Stacking Classifier: Accuracy={acc_stack:.4f}, ROC-AUC={auc_stack:.4f}")

# ============================================================================
# 6. BEST PRACTICES & RECOMMENDATIONS
# ============================================================================
print("\n" + "="*70)
print("6. ENSEMBLE METHODS BEST PRACTICES")
print("="*70)

print("""
ENSEMBLE STRATEGIES COMPARISON:

1. VOTING/AVERAGING:
   ✅ Simple, easy to understand
   ✅ Works well when base models are diverse
   ❌ All models weighted equally (unless specified)
   → Use for: Quick ensemble, combining different model types

2. BAGGING (Bootstrap Aggregating):
   ✅ Reduces variance (overfitting)
   ✅ Parallelizable (fast training)
   ✅ Good for unstable models (decision trees)
   ❌ May not reduce bias much
   → Use for: Reducing overfitting, high-variance models

3. RANDOM FOREST (Bagging + Feature Randomness):
   ✅ Bagging benefits + decorrelated trees
   ✅ Built-in feature importance
   ✅ Robust and reliable
   → Use for: General-purpose, first ensemble to try

4. BOOSTING (AdaBoost, Gradient Boosting):
   ✅ Reduces bias (underfitting)
   ✅ Often best performance
   ❌ Sequential (slower training)
   ❌ Can overfit if not tuned
   → Use for: Maximizing performance, tabular data

5. STACKING:
   ✅ Can combine very different models
   ✅ Meta-model learns optimal combination
   ❌ More complex, risk of overfitting
   ❌ Harder to interpret
   → Use for: Competitions, maximizing last % of performance

WHEN TO USE EACH:

Simple ensemble needed:
  → Voting (average 3-5 models)

High variance (overfitting):
  → Bagging or Random Forest

High bias (underfitting):
  → Boosting (GradientBoosting, XGBoost)

Maximum performance:
  → Stacking or Boosting

Limited data:
  → Bagging (reduces overfitting)

Large dataset:
  → Random Forest (parallelizable)

PRACTICAL TIPS:

1. Diversity is key:
   - Combine different model types (tree + linear)
   - Use different feature subsets
   - Vary hyperparameters

2. Start simple:
   - Try Random Forest first (bagging + built-in)
   - Then Gradient Boosting (boosting)
   - Finally stacking if needed

3. Cross-validation:
   - Always validate ensemble performance
   - Use CV to prevent overfitting

4. Computational cost:
   - Bagging: Parallelizable, fast
   - Boosting: Sequential, slower
   - Stacking: Most expensive (trains meta-model)

CHEMISTRY-SPECIFIC INSIGHTS:

For molecular property prediction:
  → Random Forest or Gradient Boosting work best
  → Stacking can squeeze out extra performance
  → Voting is good for combining different descriptor sets

For drug discovery:
  → Ensemble methods reduce false positives/negatives
  → Important when prediction errors are costly

COMMON PITFALLS:

❌ Ensembling similar models (low diversity)
❌ Not validating on separate test set
❌ Over-complicating (start simple!)
❌ Ignoring computational cost
❌ Using stacking without enough data

EXERCISE FOR YOU:
1. Try XGBoost (pip install xgboost) - often best performance
2. Experiment with weighted voting (give weights to models)
3. Create custom stacking with different meta-models
4. Compare ensemble performance vs single models on your data
5. Analyze which ensemble works best for your problem
""")

print("\n" + "="*70)
print("Activity 5 Complete!")
print("="*70)
