import random
import pandas as pd
import joblib
from collections import defaultdict

MODEL_PATH = "data/processed/world_cup_model.pkl"
DATA_PATH = "data/processed/match_features.csv"

FEATURES = [
    "home_elo_pre", "away_elo_pre", "elo_diff",
    "home_points_last5", "away_points_last5",
    "home_goals_for_last5", "away_goals_for_last5",
    "home_goals_against_last5", "away_goals_against_last5",
    "home_goal_diff_last5", "away_goal_diff_last5",
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

    home_is_home = latest_home["home_team"] == home_team
    away_is_home = latest_away["home_team"] == away_team

    home_elo = latest_home["home_elo_pre"] if home_is_home else latest_home["away_elo_pre"]
    away_elo = latest_away["home_elo_pre"] if away_is_home else latest_away["away_elo_pre"]

    row = {
        "home_elo_pre": home_elo,
        "away_elo_pre": away_elo,
        "elo_diff": home_elo - away_elo,
        "home_points_last5": latest_home["home_points_last5"] if home_is_home else latest_home["away_points_last5"],
        "away_points_last5": latest_away["home_points_last5"] if away_is_home else latest_away["away_points_last5"],
        "home_goals_for_last5": latest_home["home_goals_for_last5"] if home_is_home else latest_home["away_goals_for_last5"],
        "away_goals_for_last5": latest_away["home_goals_for_last5"] if away_is_home else latest_away["away_goals_for_last5"],
        "home_goals_against_last5": latest_home["home_goals_against_last5"] if home_is_home else latest_home["away_goals_against_last5"],
        "away_goals_against_last5": latest_away["home_goals_against_last5"] if away_is_home else latest_away["away_goals_against_last5"],
        "home_goal_diff_last5": latest_home["home_goal_diff_last5"] if home_is_home else latest_home["away_goal_diff_last5"],
        "away_goal_diff_last5": latest_away["home_goal_diff_last5"] if away_is_home else latest_away["away_goal_diff_last5"],
        "neutral": True,
    }

    return pd.DataFrame([row])[FEATURES]


def predict_probs(model, df, team_a, team_b):
    X = get_latest_team_features(df, team_a, team_b)
    probs = model.predict_proba(X)[0]

    return {
        "team_a_win": probs[2],
        "draw": probs[1],
        "team_b_win": probs[0],
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
    if result == "team_b_win":
        return 0, 3
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

    return sorted(
        table.values(),
        key=lambda x: (x["points"], x["wins"]),
        reverse=True,
    )


def simulate_knockout_match(model, df, team_a, team_b):
    probs = predict_probs(model, df, team_a, team_b)

    # Knockout games cannot end in draws.
    team_a_prob = probs["team_a_win"] + probs["draw"] / 2
    team_b_prob = probs["team_b_win"] + probs["draw"] / 2

    return random.choices(
        [team_a, team_b],
        weights=[team_a_prob, team_b_prob],
        k=1,
    )[0]


def simulate_knockout_bracket(model, df, group_winners, runners_up, best_third):
    third_teams = [team["team"] for team in best_third]

    # Simplified fixed Round of 32 bracket
    round_of_32 = [
        (group_winners[0], third_teams[0]),
        (runners_up[0], runners_up[1]),
        (group_winners[1], third_teams[1]),
        (runners_up[2], runners_up[3]),

        (group_winners[2], third_teams[2]),
        (runners_up[4], runners_up[5]),
        (group_winners[3], third_teams[3]),
        (runners_up[6], runners_up[7]),

        (group_winners[4], third_teams[4]),
        (runners_up[8], runners_up[9]),
        (group_winners[5], third_teams[5]),
        (runners_up[10], runners_up[11]),

        (group_winners[6], third_teams[6]),
        (group_winners[7], third_teams[7]),
        (group_winners[8], group_winners[9]),
        (group_winners[10], group_winners[11]),
    ]

    current_round = [
        simulate_knockout_match(model, df, team_a, team_b)
        for team_a, team_b in round_of_32
    ]

    while len(current_round) > 1:
        next_round = []

        for i in range(0, len(current_round), 2):
            winner = simulate_knockout_match(
                model,
                df,
                current_round[i],
                current_round[i + 1]
            )
            next_round.append(winner)

        current_round = next_round

    return current_round[0]


def simulate_one_tournament(model, df):
    group_winners = []
    runners_up = []
    third_place = []

    for group_name, teams in GROUPS.items():
        standings = simulate_group(model, df, group_name, teams)

        group_winners.append(standings[0]["team"])
        runners_up.append(standings[1]["team"])
        third_place.append(standings[2])

    best_third_place = sorted(
        third_place,
        key=lambda x: (x["points"], x["wins"]),
        reverse=True,
    )[:8]

    knockout_teams = (
        group_winners
        + runners_up
        + [team["team"] for team in best_third_place]
    )

    champion = simulate_knockout_bracket(
        model,
        df,
        group_winners,
        runners_up,
        best_third_place
    )

    return champion, knockout_teams


def main():
    print("Loading model and data...")
    model = joblib.load(MODEL_PATH)

    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])

    simulations = 1000

    champion_counts = defaultdict(int)
    knockout_counts = defaultdict(int)

    print("Running full tournament simulations...")

    for sim in range(simulations):
        if sim % 100 == 0:
            print(f"Simulation {sim}/{simulations}")

        champion, knockout_teams = simulate_one_tournament(model, df)

        champion_counts[champion] += 1

        for team in knockout_teams:
            knockout_counts[team] += 1

    print("\nChampion odds:")
    for team, count in sorted(champion_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{team}: {count / simulations:.1%}")

    print("\nKnockout qualification odds:")
    for team, count in sorted(knockout_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{team}: {count / simulations:.1%}")


if __name__ == "__main__":
    main()

