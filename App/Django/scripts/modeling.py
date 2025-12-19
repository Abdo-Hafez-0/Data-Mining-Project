def train_model(featured_path):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    df = pd.read_csv(featured_path)

    X = df[['attempts', 'avg_time', 'time_std']]
    y = pd.qcut(df['difficulty'], q=3, labels=[0,1,2])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = RandomForestClassifier(n_estimators=200)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))

    return {
        "accuracy": acc,
        "features": list(X.columns)
    }
