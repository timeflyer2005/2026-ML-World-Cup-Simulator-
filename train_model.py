import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, classification_report
from xgboost import XGBClassifier
import joblib

DATA_PATH = "data/processed/match_features.csv"
MODEL_PATH = "data/processed/world_cup_model.pkl"


def main():
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])

    # Remove unfinished/future matches
    df = df.dropna(subset=["home_score", "away_score"])

    features = [
        "home_elo_pre",
        "away_elo_pre",
        "elo_diff",
        "home_points_last5",
        "away_points_last5",
        "home_goals_for_last5",
        "away_goals_for_last5",
        "home_goals_against_last5",
        "away_goals_against_last5",
        "home_goal_diff_last5",
        "away_goal_diff_last5",
        "neutral",
    ]

    X = df[features]
    y = df["target"]

    # Time split: train on older matches, test on newer matches
    train_mask = df["date"] < "2022-01-01"
    test_mask = df["date"] >= "2022-01-01"

    X_train = X[train_mask]
    y_train = y[train_mask]

    X_test = X[test_mask]
    y_test = y[test_mask]

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        eval_metric="mlogloss"
    )

    print("Training model...")
    model.fit(X_train, y_train)

    print("Testing model...")
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)

    print("Accuracy:", accuracy_score(y_test, preds))
    print("Log loss:", log_loss(y_test, probs))
    print(classification_report(y_test, preds))

    joblib.dump(model, MODEL_PATH)
    print("Saved model:", MODEL_PATH)


if __name__ == "__main__":
    main()

