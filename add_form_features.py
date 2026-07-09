import pandas as pd
from collections import defaultdict, deque

INPUT_PATH = "data/processed/matches_with_elo.csv"
OUTPUT_PATH = "data/processed/match_features.csv"


def load_data():
    df = pd.read_csv(INPUT_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Keep modern matches only
    df = df[df["date"] >= "1990-01-01"].copy()

    return df


def empty_form():
    return {
        "points_last5": 0,
        "goals_for_last5": 0,
        "goals_against_last5": 0,
        "goal_diff_last5": 0,
        "matches_played_last5": 0,
    }


def summarize_last5(matches):
    if len(matches) == 0:
        return empty_form()

    points = sum(m["points"] for m in matches)
    goals_for = sum(m["goals_for"] for m in matches)
    goals_against = sum(m["goals_against"] for m in matches)
    matches_played = len(matches)

    return {
        "points_last5": points / matches_played,
        "goals_for_last5": goals_for / matches_played,
        "goals_against_last5": goals_against / matches_played,
        "goal_diff_last5": (goals_for - goals_against) / matches_played,
        "matches_played_last5": matches_played,
    }


def get_points(goals_for, goals_against):
    if goals_for > goals_against:
        return 3
    elif goals_for == goals_against:
        return 1
    else:
        return 0


def add_form_features(df):
    team_history = defaultdict(lambda: deque(maxlen=5))
    rows = []

    for _, row in df.iterrows():
        home_team = row["home_team"]
        away_team = row["away_team"]

        home_form = summarize_last5(team_history[home_team])
        away_form = summarize_last5(team_history[away_team])

        new_row = row.to_dict()

        for key, value in home_form.items():
            new_row[f"home_{key}"] = value

        for key, value in away_form.items():
            new_row[f"away_{key}"] = value

        rows.append(new_row)

        home_goals = row["home_score"]
        away_goals = row["away_score"]

        team_history[home_team].append({
            "points": get_points(home_goals, away_goals),
            "goals_for": home_goals,
            "goals_against": away_goals,
        })

        team_history[away_team].append({
            "points": get_points(away_goals, home_goals),
            "goals_for": away_goals,
            "goals_against": home_goals,
        })

    return pd.DataFrame(rows)


def main():
    print("Loading matches with Elo...")
    df = load_data()
    print("Loaded rows:", len(df))

    print("Adding last-5 form features...")
    df = add_form_features(df)
    print("Added form features")

    print("Saving final feature dataset...")
    df.to_csv(OUTPUT_PATH, index=False)

    print("Saved:", OUTPUT_PATH)
    print(df.head())
    print("Rows:", len(df))


if __name__ == "__main__":
    main()