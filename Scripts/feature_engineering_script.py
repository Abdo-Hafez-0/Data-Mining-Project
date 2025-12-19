import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# ============================================================================
# LOAD DATA
# ============================================================================

df = pd.read_csv("../Data/Cleaned/cleaned_data.csv")

print("=" * 80)
print("FEATURE ENGINEERING PIPELINE")
print("=" * 80)
print(f"\nInitial dataset shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())

# ============================================================================
# 1. INFER CORRECT ANSWERS USING MAJORITY VOTING
# ============================================================================

print("\n" + "=" * 80)
print("STEP 1: INFERRING CORRECT ANSWERS")
print("=" * 80)

# The dataset does not contain ground-truth correct answers.
# We infer them using majority voting, assuming the most selected answer
# represents the correct one.

answer_counts = df.groupby(['question_id', 'user_answer']).size()

correct_answers = (
    answer_counts
    .groupby(level=0)
    .idxmax()
    .apply(lambda x: x[1])
    .reset_index(name='assumed_correct_answer')
)

print(f"\nInferred correct answers for {len(correct_answers)} questions")
print("\nSample of inferred correct answers:")
print(correct_answers.head(10))

# ============================================================================
# 2. CONTROL LABEL NOISE (MINIMUM ATTEMPTS FILTER)
# ============================================================================

print("\n" + "=" * 80)
print("STEP 2: FILTERING QUESTIONS BY MINIMUM ATTEMPTS")
print("=" * 80)

# Questions with very few attempts produce unreliable inferred answers.
# We keep only questions with sufficient attempts.

MINIMUM_ATTEMPTS = 10

attempts_per_question = df['question_id'].value_counts()
valid_questions = attempts_per_question[attempts_per_question >= MINIMUM_ATTEMPTS].index

print(f"\nQuestions before filtering: {df['question_id'].nunique()}")
print(f"Questions with ≥{MINIMUM_ATTEMPTS} attempts: {len(valid_questions)}")

df = df[df['question_id'].isin(valid_questions)]

print(f"Dataset shape after filtering: {df.shape}")
print(f"Records removed: {df.shape[0]}")

# ============================================================================
# 3. CREATE CORRECTNESS INDICATOR (is_correct)
# ============================================================================

print("\n" + "=" * 80)
print("STEP 3: CREATING CORRECTNESS INDICATOR")
print("=" * 80)

# A binary feature indicating whether the user's answer matches
# the inferred correct answer.

df = df.merge(correct_answers, on='question_id', how='left')
df['is_correct'] = (df['user_answer'] == df['assumed_correct_answer']).astype(int)

print(f"\nCorrectness distribution:")
print(df['is_correct'].value_counts())
print(f"\nOverall success rate: {df['is_correct'].mean():.2%}")

print("\nSample of data with correctness:")
print(df[['question_id', 'user_answer', 'assumed_correct_answer', 'is_correct']].head(10))

# ============================================================================
# 4. AGGREGATE DATA TO QUESTION LEVEL
# ============================================================================

print("\n" + "=" * 80)
print("STEP 4: AGGREGATING TO QUESTION LEVEL")
print("=" * 80)

# Behavioral features are aggregated per question to reduce noise.

question_df = df.groupby('question_id').agg(
    attempts=('user_id', 'count'),
    success_rate=('is_correct', 'mean'),
    avg_time=('elapsed_time_seconds', 'mean'),
    median_time=('elapsed_time_seconds', 'median'),
    time_std=('elapsed_time_seconds', 'std')
).reset_index()

print(f"\nQuestions in aggregated dataset: {len(question_df)}")
print("\nAggregated question statistics:")
print(question_df.describe())

# ============================================================================
# 5. ENGINEER QUESTION-LEVEL FEATURES
# ============================================================================

print("\n" + "=" * 80)
print("STEP 5: ENGINEERING QUESTION-LEVEL FEATURES")
print("=" * 80)

# These features describe question behavior without encoding difficulty directly.

# Log-transformed attempts (handles skewness)
question_df['log_attempts'] = np.log1p(question_df['attempts'])

# Time skew: difference between mean and median time
# Positive values indicate right-skewed distribution (few very slow solvers)
question_df['time_skew'] = question_df['avg_time'] - question_df['median_time']

# Relative time variance: coefficient of variation
# Measures consistency of solving times
question_df['relative_time_variance'] = (
    question_df['time_std'] / question_df['avg_time']
)

print("\nEngineered features:")
print("- log_attempts: Log-transformed attempt count")
print("- time_skew: Difference between mean and median time")
print("- relative_time_variance: Coefficient of variation for time")

print("\nFeature statistics:")
print(question_df[['log_attempts', 'time_skew', 'relative_time_variance']].describe())

# ============================================================================
# 6. DEFINE DIFFICULTY TARGET (COMPOSITE SCORE)
# ============================================================================

print("\n" + "=" * 80)
print("STEP 6: DEFINING DIFFICULTY TARGET")
print("=" * 80)

# Difficulty is defined only after feature creation.
# It combines standardized success rate and time:
# - Lower success rate → higher difficulty
# - Higher time → higher difficulty

scaler = StandardScaler()

question_df[['success_z', 'time_z']] = scaler.fit_transform(
    question_df[['success_rate', 'avg_time']]
)

question_df['difficulty'] = (
    (-question_df['success_z']) + question_df['time_z']
)

print("\nDifficulty score components:")
print("- success_z: Standardized success rate (inverted)")
print("- time_z: Standardized average time")
print("- difficulty: Combined score")

print("\nDifficulty score statistics:")
print(question_df['difficulty'].describe())

# ============================================================================
# 7. CREATE DIFFICULTY CLASSES
# ============================================================================

print("\n" + "=" * 80)
print("STEP 7: CREATING DIFFICULTY CLASSES")
print("=" * 80)

# Create 3-class difficulty labels using quantile-based binning
question_df['difficulty_class'] = pd.qcut(
    question_df['difficulty'],
    q=3,    
    labels=['Easy', 'Medium', 'Hard']
)

print("\nDifficulty class distribution:")
print(question_df['difficulty_class'].value_counts().sort_index())

# Create binary difficulty for simplified classification
question_df['difficulty_binary'] = question_df['difficulty_class'].map(
    {
        'Easy': 'Easy',
        'Hard': 'Hard'
    }
)

print("\nBinary difficulty distribution (before dropping Medium):")
print(question_df['difficulty_binary'].value_counts())

# Remove Medium class (only keep Easy and Hard for binary classification)
question_df = question_df.dropna(subset=['difficulty_binary'])

print(f"\nQuestions after removing 'Medium' class: {len(question_df)}")

# Drop binary column (keeping only the 3-class version)
question_df = question_df.drop(columns=["difficulty_binary"])

# ============================================================================
# 8. FINAL DATASET SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("FINAL FEATURE-ENRICHED DATASET")
print("=" * 80)

print(f"\nFinal dataset shape: {question_df.shape}")
print(f"\nColumns: {list(question_df.columns)}")

print("\nFirst 10 rows of final dataset:")
print(question_df.head(10))

print("\nFinal statistics:")
print(question_df.describe())

# ============================================================================
# 9. SAVE FEATURE-ENRICHED DATASET
# ============================================================================

print("\n" + "=" * 80)
print("SAVING DATASET")
print("=" * 80)

output_path = "../Data/Featured/featured_data.csv"
question_df.to_csv(output_path, index=False)

print(f"\nFeature-enriched dataset saved to: {output_path}")
print(f"Total features: {len(question_df.columns)}")
print(f"Total questions: {len(question_df)}")

print("\n" + "=" * 80)
print("FEATURE ENGINEERING COMPLETE")
print("=" * 80)