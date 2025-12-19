def run_cleaning(input_path, output_path):
    import pandas as pd

    df = pd.read_csv(input_path)

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['elapsed_time_seconds'] = df['elapsed_time'] / 1000
    df.drop(columns=['elapsed_time'], inplace=True)

    df.dropna(inplace=True)
    df = df.drop_duplicates(
        subset=['user_id', 'question_id', 'timestamp']
    )

    df = df[df['elapsed_time_seconds'] >= 1]
    upper = df['elapsed_time_seconds'].quantile(0.99)
    df = df[df['elapsed_time_seconds'] <= upper]

    df = df.sort_values(['user_id', 'timestamp'])
    df.to_csv(output_path, index=False)

    return {
        "rows_after": len(df),
        "columns": list(df.columns)
    }
