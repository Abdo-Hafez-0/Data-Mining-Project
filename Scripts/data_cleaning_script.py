import pandas as pd
import numpy as np

# Load the raw data
df = pd.read_csv("../Data/Raw/Raw_Data.csv")

# 1.1 Convert Timestamp
# The timestamp column is stored as Unix time in milliseconds
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

# 1.2 Convert Elapsed Time to Seconds
# Elapsed time is converted from milliseconds to seconds
df['elapsed_time_seconds'] = df['elapsed_time'] / 1000
df.drop(columns=['elapsed_time'], inplace=True)

# 1.3 Ensure Correct String Types
df['question_id'] = df['question_id'].astype(str)
df['user_id'] = df['user_id'].astype(str)
df['user_answer'] = df['user_answer'].astype(str)

# 2. Handle Missing Values
# Missing values in core behavioral fields cannot be inferred reliably
df.dropna(inplace=True)

# 3. Remove Duplicate Interactions
# Duplicate records artificially inflate attempts and bias difficulty estimation
# A duplicate is defined as:
# - Same user
# - Same question
# - Same timestamp
df = df.drop_duplicates(subset=['user_id', 'question_id', 'timestamp'])

# 4. Standardize User Answers
# User answers may contain inconsistent formatting
df['user_answer'] = df['user_answer'].str.lower().str.strip()

# 5. Handle Invalid and Extreme Time Values
# 5.1 Remove Zero or Negative Time Values
df = df[df['elapsed_time_seconds'] > 0]

# 5.2 Remove Unrealistically Fast Answers
# Answers faster than 1 second are considered invalid
df = df[df['elapsed_time_seconds'] >= 1]

# 5.3 Remove Extreme Slow Outliers (Top 1%)
# Very large time values introduce noise
upper_limit = df['elapsed_time_seconds'].quantile(0.99)
df = df[df['elapsed_time_seconds'] <= upper_limit]

# 6. Validate User and Question IDs
# Ensure IDs follow the expected format
# 6.1 Validate Question IDs
df = df[df['question_id'].str.startswith('q')]

# 6.2 Validate User IDs
df = df[df['user_id'].str.startswith('u')]

# 7. Sort Dataset for Behavioral Analysis
# Sorting is required for later feature engineering steps
df = df.sort_values(['user_id', 'timestamp'])

# Display summary statistics
print("Dataset Info:")
print(df.info())
print("\nDataset Description:")
print(df.describe())
print("\nUnique Values:")
print(df.nunique())
print("\nFirst few rows:")
print(df.head())

# 8. Save Cleaned Dataset
df.to_csv("../Data/Cleaned/cleaned_data.csv", index=False)
print("\nCleaned data saved to ../Data/Cleaned/cleaned_data.csv")