import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# 1. LOAD AND INSPECT DATA
# ============================================================================

df = pd.read_csv("../Data/Cleaned/cleaned_data.csv")

print("=" * 80)
print("DATASET OVERVIEW")
print("=" * 80)
print("\nFirst 10 rows:")
print(df.head(10))
print("\nDataset Info:")
print(df.info())
print("\nDescriptive Statistics:")
print(df.describe())

# ============================================================================
# 2. TEMPORAL FEATURE ENGINEERING
# ============================================================================

# Convert timestamp to datetime with proper format
df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', utc=True)

# Extract temporal features
df['hour'] = df['timestamp'].dt.hour
df['day_name'] = df['timestamp'].dt.day_name()
df['date'] = df['timestamp'].dt.date

print("\n" + "=" * 80)
print("TEMPORAL ANALYSIS")
print("=" * 80)
print("\nSolving activity by hour:")
print(df.groupby('hour').size())

# ============================================================================
# 3. TIME DISTRIBUTION ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("ELAPSED TIME ANALYSIS")
print("=" * 80)
print("\nElapsed time statistics:")
print(df['elapsed_time_seconds'].describe())
print("\nQuantiles (90%, 95%, 99%):")
print(df['elapsed_time_seconds'].quantile([0.9, 0.95, 0.99]))

# Plot full distribution
plt.figure(figsize=(12, 5))
plt.hist(df['elapsed_time_seconds'], bins=200)
plt.xlabel("Elapsed Time (sec)")
plt.ylabel("Count")
plt.title("Elapsed Time Distribution (Full Range)")
plt.tight_layout()
plt.show()

# Apply threshold for focused analysis
THRESHOLD = 120
df_eda = df[df['elapsed_time_seconds'] <= THRESHOLD].copy()
print(f"\nRecords after applying {THRESHOLD}s threshold: {len(df_eda)}")

# Plot filtered distribution
plt.figure(figsize=(12, 5))
plt.hist(df_eda['elapsed_time_seconds'], bins=100)
plt.xlabel("Elapsed Time (sec)")
plt.ylabel("Count")
plt.title(f"Elapsed Time Distribution (≤{THRESHOLD} sec)")
plt.tight_layout()
plt.show()

# ============================================================================
# 4. USER BEHAVIOR ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("USER BEHAVIOR ANALYSIS")
print("=" * 80)

# User attempts analysis
user_attempts = df_eda.groupby('user_id').size()
print("\nAttempts per user - Statistics:")
print(user_attempts.describe())
print("\nTop 10 most active users:")
print(user_attempts.sort_values(ascending=False).head(10))

plt.figure(figsize=(12, 5))
plt.hist(user_attempts, bins=100)
plt.xlabel("Attempts per User")
plt.ylabel("Number of Users")
plt.title("Distribution of User Activity")
plt.tight_layout()
plt.show()

# User speed analysis
user_speed = df_eda.groupby('user_id')['elapsed_time_seconds'].mean()
print("\nUser speed statistics:")
print(user_speed.describe())

# ============================================================================
# 5. QUESTION DIFFICULTY ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("QUESTION DIFFICULTY ANALYSIS")
print("=" * 80)

# Calculate question statistics
question_stats = df_eda.groupby('question_id')['elapsed_time_seconds'].agg(['mean', 'median', 'count'])

# Filter questions with at least 5 attempts for reliable statistics
question_stats_clean = question_stats[question_stats['count'] >= 5]
question_stats_clean = question_stats_clean.sort_values('mean', ascending=False)

print("\nTop 10 most difficult questions (by mean time):")
print(question_stats_clean.head(10))

plt.figure(figsize=(12, 6))
sns.histplot(question_stats_clean['mean'], bins=50, kde=True)
plt.xlabel("Mean Elapsed Time per Question (sec)")
plt.ylabel("Number of Questions")
plt.title("Distribution of Question Difficulty (Questions with ≥5 attempts)")
plt.tight_layout()
plt.show()

# ============================================================================
# 6. ANSWER DISTRIBUTION ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("ANSWER DISTRIBUTION")
print("=" * 80)
print("\nUser answer distribution (normalized):")
print(df['user_answer'].value_counts(normalize=True))

# ============================================================================
# 7. CORRELATION ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("CORRELATION ANALYSIS")
print("=" * 80)
print("\nCorrelation between elapsed time and hour of day:")
print(df_eda[['elapsed_time_seconds', 'hour']].corr())

# ============================================================================
# 8. USER PROFILING AND SEGMENTATION
# ============================================================================

print("\n" + "=" * 80)
print("USER PROFILING AND SEGMENTATION")
print("=" * 80)

# Create user profile dataframe
user_profile = pd.concat([user_attempts, user_speed], axis=1)
user_profile.columns = ['attempts', 'avg_time']

# Plot average time distribution
plt.figure(figsize=(10, 5))
plt.hist(user_profile['avg_time'], bins=50)
plt.xlabel("Average Time per Question (sec)")
plt.ylabel("Number of Users")
plt.title("Distribution of User Speed")
plt.tight_layout()
plt.show()

# Segmentation by activity level
def attempt_segment(x):
    if x < 30:
        return "Casual"
    elif x < 100:
        return "Normal"
    elif x < 500:
        return "Heavy"
    else:
        return "Extreme"

# Segmentation by speed
def speed_segment(x):
    if x < 20:
        return "Fast"
    elif x < 30:
        return "Normal"
    else:
        return "Slow"

user_profile['attempt_segment'] = user_profile['attempts'].apply(attempt_segment)
user_profile['speed_segment'] = user_profile['avg_time'].apply(speed_segment)

print("\nUser segments by activity:")
print(user_profile['attempt_segment'].value_counts())
print("\nUser segments by speed:")
print(user_profile['speed_segment'].value_counts())

# Visualize user profiling
plt.figure(figsize=(12, 6))
sns.scatterplot(data=user_profile, x='attempts', y='avg_time', 
                hue='attempt_segment', style='speed_segment', s=50, alpha=0.6)
plt.xlabel("Attempts per User")
plt.ylabel("Average Time per Question (sec)")
plt.title("User Profiling: Activity vs Speed")
plt.xscale('log')  # Log scale for better visualization of long-tail distribution
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# ============================================================================
# 9. TEMPORAL PATTERN ANALYSIS (HEATMAP)
# ============================================================================

print("\n" + "=" * 80)
print("TEMPORAL PATTERN ANALYSIS")
print("=" * 80)

# Create heatmap of average elapsed time by day and hour
heatmap_data = df_eda.pivot_table(index='day_name', columns='hour', 
                                   values='elapsed_time_seconds', aggfunc='mean')

# Reorder days for proper display
days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
heatmap_data = heatmap_data.reindex(days_order)

plt.figure(figsize=(15, 6))
sns.heatmap(heatmap_data, cmap="YlGnBu", annot=False, fmt='.1f', cbar_kws={'label': 'Avg Time (sec)'})
plt.xlabel("Hour of Day")
plt.ylabel("Day of Week")
plt.title("Average Elapsed Time by Day and Hour")
plt.tight_layout()
plt.show()

# Activity count heatmap
activity_heatmap = df_eda.pivot_table(index='day_name', columns='hour', 
                                       values='elapsed_time_seconds', aggfunc='count')
activity_heatmap = activity_heatmap.reindex(days_order)

plt.figure(figsize=(15, 6))
sns.heatmap(activity_heatmap, cmap="YlOrRd", annot=False, fmt='g', cbar_kws={'label': 'Count'})
plt.xlabel("Hour of Day")
plt.ylabel("Day of Week")
plt.title("User Activity Count by Day and Hour")
plt.tight_layout()
plt.show()

print("\n" + "=" * 80)
print("EDA COMPLETE")
print("=" * 80)