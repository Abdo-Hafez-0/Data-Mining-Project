import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# =====================================================
# 1️⃣ Data Cleaning
# =====================================================
def run_cleaning(input_path, output_path):
    df = pd.read_csv(input_path)

    # Convert timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    # Convert elapsed time to seconds
    df['elapsed_time_seconds'] = df['elapsed_time'] / 1000
    df.drop(columns=['elapsed_time'], inplace=True)

    # Remove missing values
    df.dropna(inplace=True)

    # Remove duplicates
    df = df.drop_duplicates(
        subset=['user_id', 'question_id', 'timestamp']
    )

    # Remove unrealistically fast answers
    df = df[df['elapsed_time_seconds'] >= 1]

    # Remove extreme outliers (top 1%)
    upper = df['elapsed_time_seconds'].quantile(0.99)
    df = df[df['elapsed_time_seconds'] <= upper]

    # Sort for consistency
    df = df.sort_values(['user_id', 'timestamp'])

    # Save cleaned data
    df.to_csv(output_path, index=False)

    return {
        "rows_after": len(df),
        "columns": list(df.columns)
    }


# =====================================================
# 2️⃣ Exploratory Data Analysis (EDA)
# =====================================================
def run_eda(cleaned_path):
    df = pd.read_csv(cleaned_path)

    return {
        "rows": len(df),
        "avg_time": df['elapsed_time_seconds'].mean(),
        "max_time": df['elapsed_time_seconds'].max(),
        "min_time": df['elapsed_time_seconds'].min()
    }


# =====================================================
# 3️⃣ Feature Engineering
# =====================================================
def run_feature_engineering(input_path, output_path):
    df = pd.read_csv(input_path)

    # Most common answer per question (assumed correct)
    answer_counts = df.groupby(
        ['question_id', 'user_answer']
    ).size()

    correct_answers = (
        answer_counts
        .groupby(level=0)
        .idxmax()
        .apply(lambda x: x[1])
        .reset_index(name='assumed_correct_answer')
    )

    df = df.merge(correct_answers, on='question_id', how='left')

    # Binary correctness
    df['is_correct'] = (
        df['user_answer'] == df['assumed_correct_answer']
    ).astype(int)

    # Aggregate question-level features
    question_df = df.groupby('question_id').agg(
        attempts=('user_id', 'count'),
        success_rate=('is_correct', 'mean'),
        avg_time=('elapsed_time_seconds', 'mean'),
        median_time=('elapsed_time_seconds', 'median'),
        time_std=('elapsed_time_seconds', 'std')
    ).reset_index()

    # Handle NaN std (single-attempt questions)
    question_df['time_std'] = question_df['time_std'].fillna(0)

    # Standardize success & time
    scaler = StandardScaler()
    question_df[['success_z', 'time_z']] = scaler.fit_transform(
        question_df[['success_rate', 'avg_time']]
    )

    # Difficulty score
    question_df['difficulty'] = (
        -question_df['success_z'] + question_df['time_z']
    )

    # Save features
    question_df.to_csv(output_path, index=False)

    return {
        "questions": len(question_df),
        "features": list(question_df.columns)
    }


# =====================================================
# 4️⃣ Model Training
# =====================================================
def train_model(featured_path):
    df = pd.read_csv(featured_path)

    # Features & target
    X = df[['attempts', 'avg_time', 'time_std']]
    y = pd.qcut(df['difficulty'], q=3, labels=[0, 1, 2])  # Easy, Medium, Hard

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    # Model
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluation
    accuracy = accuracy_score(y_test, model.predict(X_test))

    return {
        "accuracy": accuracy,
        "features": list(X.columns)
    }
