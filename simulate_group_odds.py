import random
import pandas as pd
import joblib
from collections import defaultdict

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

GROUPS = {
    "A": ["Mexico", "South Africa", "South Korea", "Czechoslovakia"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curaçao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}


def get_latest_team_features(df, home_team, away_team):
    home_matches = df[(df["home_team"] == home_team) | (df["away_team"] == home_team)]
    away_matches = df[(df["home_team"] == away_team) | (df["away_team"] == away_team)]

    if home_matches.empty:
        raise ValueError(f"Team not found: {home_team}")
    if away_matches.empty:
        raise ValueError(f"Team not found: {away_team}")

    latest_home = home_matches.sort_values("date").iloc[-1]
    latest_away = away_matches.sort_values("date").iloc[-1]

    home_elo = (
        latest_home["home_elo_pre"]
        if latest_home["home_team"] == home_team
        else latest_home["away_elo_pre"]
    )
    away_elo = (
        latest_away["home_elo_pre"]
        if latest_away["home_team"] == away_team
        else latest_away["away_elo_pre"]
    )

    row = {
        "home_elo_pre": home_elo,
        "away_elo_pre": away_elo,
        "elo_diff": home_elo - away_elo,
        "home_points_last5": (
            latest_home["home_points_last5"]
            if latest_home["home_team"] == home_team
            else latest_home["away_points_last5"]
        ),
        "away_points_last5": (
            latest_away["home_points_last5"]
            if latest_away["home_team"] == away_team
            else latest_away["away_points_last5"]
        ),
        "home_goals_for_last5": (
            latest_home["home_goals_for_last5"]
            if latest_home["home_team"] == home_team
            else latest_home["away_goals_for_last5"]
        ),
        "away_goals_for_last5": (
            latest_away["home_goals_for_last5"]
            if latest_away["home_team"] == away_team
            else latest_away["away_goals_for_last5"]
        ),
        "home_goals_against_last5": (
            latest_home["home_goals_against_last5"]
            if latest_home["home_team"] == home_team
            else latest_home["away_goals_against_last5"]
        ),
        "away_goals_against_last5": (
            latest_away["home_goals_against_last5"]
            if latest_away["home_team"] == away_team
            else latest_away["away_goals_against_last5"]
        ),
        "home_goal_diff_last5": (
            latest_home["home_goal_diff_last5"]
            if latest_home["home_team"] == home_team
            else latest_home["away_goal_diff_last5"]
        ),
        "away_goal_diff_last5": (
            latest_away["home_goal_diff_last5"]
            if latest_away["home_team"] == away_team
            else latest_away["away_goal_diff_last5"]
        ),
        "neutral": True,
    }

    return pd.DataFrame([row])[FEATURES]


def predict_probs(model, df, team_a, team_b):
    X = get_latest_team_features(df, team_a, team_b)
    probs = model.predict_proba(X)[0]

    return {
        "team_a_win": probs[2],  # home win
        "draw": probs[1],
        "team_b_win": probs[0],  # away win
    }


def simulate_group_match(model, df, team_a, team_b):
    probs = predict_probs(model, df, team_a, team_b)

    result = random.choices(
        ["team_a_win", "draw", "team_b_win"],
        weights=[probs["team_a_win"], probs["draw"], probs["team_b_win"]],
        k=1,
    )[0]

    if result == "team_a_win":
        return 3, 0
    elif result == "team_b_win":
        return 0, 3
    else:
        return 1, 1


def simulate_group(model, df, group_name, teams):
    table = {
        team: {
            "team": team,
            "group": group_name,
            "points": 0,
            "wins": 0,
            "draws": 0,
            "losses": 0,
        }
        for team in teams
    }

    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            team_a = teams[i]
            team_b = teams[j]

            points_a, points_b = simulate_group_match(model, df, team_a, team_b)

            table[team_a]["points"] += points_a
            table[team_b]["points"] += points_b

            if points_a == 3:
                table[team_a]["wins"] += 1
                table[team_b]["losses"] += 1
            elif points_b == 3:
                table[team_b]["wins"] += 1
                table[team_a]["losses"] += 1
            else:
                table[team_a]["draws"] += 1
                table[team_b]["draws"] += 1

    standings = sorted(
        table.values(),
        key=lambda x: (x["points"], x["wins"]),
        reverse=True,
    )

    return standings


def main():
    print("Loading model and data...")
    model = joblib.load(MODEL_PATH)

    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])

    simulations = 1000
    advance_counts = defaultdict(int)
    group_win_counts = defaultdict(int)

    print("Running simulations...")
    for sim in range(simulations):
        if sim % 10 == 0:
            print(f"Simulation {sim}/{simulations}")

        for group_name, teams in GROUPS.items():
            standings = simulate_group(model, df, group_name, teams)

            group_winner = standings[0]["team"]
            top_two = [standings[0]["team"], standings[1]["team"]]

            group_win_counts[group_winner] += 1

            for team in top_two:
                advance_counts[team] += 1

    print("\nGroup advancement odds:")
    for group_name, teams in GROUPS.items():
        print(f"\nGroup {group_name}")
        for team in teams:
            advance_prob = advance_counts[team] / simulations
            win_group_prob = group_win_counts[team] / simulations
            print(f"{team}: advance {advance_prob:.1%}, win group {win_group_prob:.1%}")


if __name__ == "__main__":
    main()