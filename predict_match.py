import sys
import joblib
import pandas as pd

MODEL_PATH = "data/processed/world_cup_model.pkl"
DATA_PATH = "data/processed/match_features.csv"


FEATURES = [
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


def get_latest_team_features(df, home_team, away_team):
    latest_home = df[(df["home_team"] == home_team) | (df["away_team"] == home_team)].sort_values("date").iloc[-1]
    latest_away = df[(df["home_team"] == away_team) | (df["away_team"] == away_team)].sort_values("date").iloc[-1]

    home_elo = latest_home["home_elo_pre"] if latest_home["home_team"] == home_team else latest_home["away_elo_pre"]
    away_elo = latest_away["home_elo_pre"] if latest_away["home_team"] == away_team else latest_away["away_elo_pre"]

    row = {
        "home_elo_pre": home_elo,
        "away_elo_pre": away_elo,
        "elo_diff": home_elo - away_elo,

        "home_points_last5": latest_home["home_points_last5"] if latest_home["home_team"] == home_team else latest_home["away_points_last5"],
        "away_points_last5": latest_away["home_points_last5"] if latest_away["home_team"] == away_team else latest_away["away_points_last5"],

        "home_goals_for_last5": latest_home["home_goals_for_last5"] if latest_home["home_team"] == home_team else latest_home["away_goals_for_last5"],
        "away_goals_for_last5": latest_away["home_goals_for_last5"] if latest_away["home_team"] == away_team else latest_away["away_goals_for_last5"],

        "home_goals_against_last5": latest_home["home_goals_against_last5"] if latest_home["home_team"] == home_team else latest_home["away_goals_against_last5"],
        "away_goals_against_last5": latest_away["home_goals_against_last5"] if latest_away["home_team"] == away_team else latest_away["away_goals_against_last5"],

        "home_goal_diff_last5": latest_home["home_goal_diff_last5"] if latest_home["home_team"] == home_team else latest_home["away_goal_diff_last5"],
        "away_goal_diff_last5": latest_away["home_goal_diff_last5"] if latest_away["home_team"] == away_team else latest_away["away_goal_diff_last5"],

        "neutral": True,
    }

    return pd.DataFrame([row])[FEATURES]


def main():
    if len(sys.argv) < 3:
        print('Usage: python3 predict_match.py "Argentina" "France"')
        return

    home_team = sys.argv[1]
    away_team = sys.argv[2]

    model = joblib.load(MODEL_PATH)

    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])

    X = get_latest_team_features(df, home_team, away_team)
    probs = model.predict_proba(X)[0]

    print(f"\nPrediction: {home_team} vs {away_team}")
    print(f"{away_team} win: {probs[0]:.1%}")
    print(f"Draw: {probs[1]:.1%}")
    print(f"{home_team} win: {probs[2]:.1%}")


if __name__ == "__main__":
    main()