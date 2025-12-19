def run_eda(cleaned_path):
    import pandas as pd

    df = pd.read_csv(cleaned_path)

    return {
        "rows": len(df),
        "avg_time": df['elapsed_time_seconds'].mean(),
        "max_time": df['elapsed_time_seconds'].max(),
        "min_time": df['elapsed_time_seconds'].min()
    }
