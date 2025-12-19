import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, accuracy_score, precision_score, 
    recall_score, f1_score, confusion_matrix
)
import seaborn as sns

# ============================================================================
# 1. LOAD FEATURED DATA
# ============================================================================

print("=" * 80)
print("MODEL TRAINING AND EVALUATION PIPELINE")
print("=" * 80)

df = pd.read_csv("../data/featured/featured_data.csv")

print(f"\nDataset shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())

print("\nDataset info:")
print(df.info())

# ============================================================================
# 2. DEFINE TARGET VARIABLE
# ============================================================================

print("\n" + "=" * 80)
print("STEP 1: DEFINING TARGET VARIABLE")
print("=" * 80)

# We choose classification as it is more interpretable
# for educational difficulty analysis.

y = df['difficulty_class']

print(f"\nTarget variable: difficulty_class")
print(f"\nClass distribution:")
print(y.value_counts())
print(f"\nClass proportions:")
print(y.value_counts(normalize=True))

# ============================================================================
# 3. PREVENT DATA LEAKAGE - REMOVE PROBLEMATIC COLUMNS
# ============================================================================

print("\n" + "=" * 80)
print("STEP 2: REMOVING LEAKAGE-PRONE COLUMNS")
print("=" * 80)

# The following columns are removed intentionally:
# - Identifiers: question_id
# - Target construction components: success_rate, avg_time, success_z, time_z
# - Target columns: difficulty, difficulty_class

leakage_columns = [
    'question_id',      # Identifier
    'success_rate',     # Direct component of difficulty
    'avg_time',         # Direct component of difficulty
    'success_z',        # Standardized success rate
    'time_z',           # Standardized time
    'difficulty',       # Continuous target
    'difficulty_class'  # Categorical target
]

print(f"\nColumns before removal: {list(df.columns)}")
print(f"\nRemoving: {leakage_columns}")

X = df.drop(columns=leakage_columns)

print(f"\nColumns after removal: {list(X.columns)}")

# ============================================================================
# 4. SELECT FINAL FEATURE SET
# ============================================================================

print("\n" + "=" * 80)
print("STEP 3: SELECTING FINAL FEATURES")
print("=" * 80)

# Remaining features describe question behavior only:
# - attempts: Total number of attempts
# - log_attempts: Log-transformed attempts (handles skewness)
# - median_time: Median solving time (robust to outliers)
# - time_std: Standard deviation of solving times
# - time_skew: Difference between mean and median time
# - relative_time_variance: Coefficient of variation

final_features = [
    'attempts',
    'log_attempts',
    'median_time',
    'time_std',
    'time_skew',
    'relative_time_variance'
]

X = X[final_features]

print(f"\nFinal features selected: {final_features}")
print(f"\nFeature set shape: {X.shape}")
print("\nFeature statistics:")
print(X.describe())

# ============================================================================
# 5. ENCODE TARGET LABELS
# ============================================================================

print("\n" + "=" * 80)
print("STEP 4: ENCODING TARGET LABELS")
print("=" * 80)

le = LabelEncoder()
y_encoded = le.fit_transform(y)

print(f"\nLabel mapping:")
for idx, label in enumerate(le.classes_):
    print(f"  {label} -> {idx}")

print(f"\nEncoded target distribution:")
unique, counts = np.unique(y_encoded, return_counts=True)
for val, count in zip(unique, counts):
    print(f"  Class {val}: {count} samples ({count/len(y_encoded):.1%})")

# ============================================================================
# 6. TRAIN / TEST SPLIT
# ============================================================================

print("\n" + "=" * 80)
print("STEP 5: SPLITTING DATA")
print("=" * 80)

TEST_SIZE = 0.2
RANDOM_STATE = 42

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_encoded
)

print(f"\nTrain set size: {X_train.shape[0]} ({(1-TEST_SIZE)*100:.0f}%)")
print(f"Test set size: {X_test.shape[0]} ({TEST_SIZE*100:.0f}%)")

print(f"\nTrain set class distribution:")
unique, counts = np.unique(y_train, return_counts=True)
for val, count in zip(unique, counts):
    print(f"  Class {val}: {count} samples ({count/len(y_train):.1%})")

# ============================================================================
# 7. MODEL 1: LOGISTIC REGRESSION (BASELINE)
# ============================================================================

print("\n" + "=" * 80)
print("MODEL 1: LOGISTIC REGRESSION (BASELINE)")
print("=" * 80)

lr = LogisticRegression(
    max_iter=1000,
    multi_class='auto',
    random_state=RANDOM_STATE
)

print("\nTraining Logistic Regression...")
lr.fit(X_train, y_train)
print("Training complete.")

y_pred_lr = lr.predict(X_test)

print("\n" + "-" * 80)
print("LOGISTIC REGRESSION RESULTS")
print("-" * 80)
print(classification_report(y_test, y_pred_lr, target_names=le.classes_))

# ============================================================================
# 8. MODEL 2: RANDOM FOREST CLASSIFIER
# ============================================================================

print("\n" + "=" * 80)
print("MODEL 2: RANDOM FOREST CLASSIFIER")
print("=" * 80)

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    min_samples_leaf=10,
    random_state=RANDOM_STATE
)

print("\nTraining Random Forest...")
rf.fit(X_train, y_train)
print("Training complete.")

y_pred_rf = rf.predict(X_test)

print("\n" + "-" * 80)
print("RANDOM FOREST RESULTS")
print("-" * 80)
print(classification_report(y_test, y_pred_rf, target_names=le.classes_))

# ============================================================================
# 9. MODEL 3: GRADIENT BOOSTING CLASSIFIER
# ============================================================================

print("\n" + "=" * 80)
print("MODEL 3: GRADIENT BOOSTING CLASSIFIER")
print("=" * 80)

gb = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    random_state=RANDOM_STATE
)

print("\nTraining Gradient Boosting...")
gb.fit(X_train, y_train)
print("Training complete.")

y_pred_gb = gb.predict(X_test)

print("\n" + "-" * 80)
print("GRADIENT BOOSTING RESULTS")
print("-" * 80)
print(classification_report(y_test, y_pred_gb, target_names=le.classes_))

# ============================================================================
# 10. MODEL COMPARISON
# ============================================================================

print("\n" + "=" * 80)
print("MODEL COMPARISON")
print("=" * 80)

results = []
models = {
    "Logistic Regression": y_pred_lr,
    "Random Forest": y_pred_rf,
    "Gradient Boosting": y_pred_gb
}

for model_name, y_pred in models.items():
    results.append({
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average='weighted'),
        "Recall": recall_score(y_test, y_pred, average='weighted'),
        "F1 Score": f1_score(y_test, y_pred, average='weighted')
    })

results_df = pd.DataFrame(results)
print("\nPerformance Comparison:")
print(results_df.to_string(index=False))

# Find best model
best_model_idx = results_df['F1 Score'].idxmax()
best_model_name = results_df.loc[best_model_idx, 'Model']
print(f"\n*** Best Model: {best_model_name} ***")

# ============================================================================
# 11. CROSS-VALIDATION (BEST MODEL)
# ============================================================================

print("\n" + "=" * 80)
print("CROSS-VALIDATION EVALUATION")
print("=" * 80)

# Cross-validation ensures the model generalizes beyond the test split.

print(f"\nPerforming 5-fold cross-validation on {best_model_name}...")

cv_scores = cross_val_score(
    gb,
    X,
    y_encoded,
    cv=5,
    scoring='f1_weighted'
)

print(f"\nCross-validation F1 scores: {cv_scores}")
print(f"Mean CV F1 score: {cv_scores.mean():.4f}")
print(f"Std CV F1 score: {cv_scores.std():.4f}")

# ============================================================================
# 12. FEATURE IMPORTANCE ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("FEATURE IMPORTANCE ANALYSIS")
print("=" * 80)

importances = pd.Series(
    gb.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nFeature Importance (Gradient Boosting):")
for feature, importance in importances.items():
    print(f"  {feature}: {importance:.4f}")

# Visualize feature importance
plt.figure(figsize=(10, 6))
importances.plot(kind='bar', color='steelblue')
plt.title("Feature Importance (Gradient Boosting)", fontsize=14, fontweight='bold')
plt.xlabel("Features", fontsize=12)
plt.ylabel("Importance", fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# ============================================================================
# 13. CONFUSION MATRIX VISUALIZATION
# ============================================================================

print("\n" + "=" * 80)
print("CONFUSION MATRIX (BEST MODEL)")
print("=" * 80)

cm = confusion_matrix(y_test, y_pred_gb)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Confusion Matrix - Gradient Boosting", fontsize=14, fontweight='bold')
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)
plt.tight_layout()
plt.show()

# ============================================================================
# 14. SAVE BEST MODEL
# ============================================================================

print("\n" + "=" * 80)
print("SAVING MODEL")
print("=" * 80)

model_path = "../models/difficulty_model.pkl"
encoder_path = "../models/label_encoder.pkl"

print(f"\nSaving Gradient Boosting model to: {model_path}")
joblib.dump(gb, model_path)

print(f"Saving Label Encoder to: {encoder_path}")
joblib.dump(le, encoder_path)

print("\n*** Model artifacts saved successfully ***")

# ============================================================================
# 15. FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("TRAINING PIPELINE COMPLETE")
print("=" * 80)

print(f"\nBest Model: {best_model_name}")
print(f"Test F1 Score: {results_df.loc[best_model_idx, 'F1 Score']:.4f}")
print(f"Cross-Validation F1 Score: {cv_scores.mean():.4f}")
print(f"\nModel saved to: {model_path}")
print(f"Encoder saved to: {encoder_path}")

print("\n" + "=" * 80)