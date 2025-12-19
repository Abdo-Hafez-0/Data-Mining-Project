def run_feature_engineering(input_path, output_path):
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler

    df = pd.read_csv(input_path)

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
    df['is_correct'] = (
        df['user_answer'] == df['assumed_correct_answer']
    ).astype(int)

    question_df = df.groupby('question_id').agg(
        attempts=('user_id', 'count'),
        success_rate=('is_correct', 'mean'),
        avg_time=('elapsed_time_seconds', 'mean'),
        median_time=('elapsed_time_seconds', 'median'),
        time_std=('elapsed_time_seconds', 'std')
    ).reset_index()

    scaler = StandardScaler()
    question_df[['success_z', 'time_z']] = scaler.fit_transform(
        question_df[['success_rate', 'avg_time']]
    )

    question_df['difficulty'] = (
        -question_df['success_z'] + question_df['time_z']
    )

    question_df.to_csv(output_path, index=False)

    return {
        "questions": len(question_df),
        "features": list(question_df.columns)
    }
