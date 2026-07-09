import random
from collections import defaultdict

import joblib
import pandas as pd
import streamlit as st

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


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df


def get_latest_team_features(df, team_a, team_b):
    a_matches = df[(df["home_team"] == team_a) | (df["away_team"] == team_a)]
    b_matches = df[(df["home_team"] == team_b) | (df["away_team"] == team_b)]

    latest_a = a_matches.sort_values("date").iloc[-1]
    latest_b = b_matches.sort_values("date").iloc[-1]

    a_is_home = latest_a["home_team"] == team_a
    b_is_home = latest_b["home_team"] == team_b

    a_elo = latest_a["home_elo_pre"] if a_is_home else latest_a["away_elo_pre"]
    b_elo = latest_b["home_elo_pre"] if b_is_home else latest_b["away_elo_pre"]

    row = {
        "home_elo_pre": a_elo,
        "away_elo_pre": b_elo,
        "elo_diff": a_elo - b_elo,
        "home_points_last5": latest_a["home_points_last5"] if a_is_home else latest_a["away_points_last5"],
        "away_points_last5": latest_b["home_points_last5"] if b_is_home else latest_b["away_points_last5"],
        "home_goals_for_last5": latest_a["home_goals_for_last5"] if a_is_home else latest_a["away_goals_for_last5"],
        "away_goals_for_last5": latest_b["home_goals_for_last5"] if b_is_home else latest_b["away_goals_for_last5"],
        "home_goals_against_last5": latest_a["home_goals_against_last5"] if a_is_home else latest_a["away_goals_against_last5"],
        "away_goals_against_last5": latest_b["home_goals_against_last5"] if b_is_home else latest_b["away_goals_against_last5"],
        "home_goal_diff_last5": latest_a["home_goal_diff_last5"] if a_is_home else latest_a["away_goal_diff_last5"],
        "away_goal_diff_last5": latest_b["home_goal_diff_last5"] if b_is_home else latest_b["away_goal_diff_last5"],
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

    return sorted(
        table.values(),
        key=lambda x: (x["points"], x["wins"]),
        reverse=True,
    )


def simulate_knockout_match(model, df, team_a, team_b):
    probs = predict_probs(model, df, team_a, team_b)

    team_a_prob = probs["team_a_win"] + probs["draw"] / 2
    team_b_prob = probs["team_b_win"] + probs["draw"] / 2

    return random.choices(
        [team_a, team_b],
        weights=[team_a_prob, team_b_prob],
        k=1,
    )[0]


def simulate_knockout_bracket(model, df, group_winners, runners_up, best_third):
    third_teams = [team["team"] for team in best_third]

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
        simulate_knockout_match(model, df, a, b)
        for a, b in round_of_32
    ]

    while len(current_round) > 1:
        next_round = []

        for i in range(0, len(current_round), 2):
            winner = simulate_knockout_match(
                model,
                df,
                current_round[i],
                current_round[i + 1],
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
        best_third_place,
    )

    return champion, knockout_teams


def run_tournament_simulations(model, df, simulations):
    champion_counts = defaultdict(int)
    knockout_counts = defaultdict(int)

    progress = st.progress(0)

    for sim in range(simulations):
        champion, knockout_teams = simulate_one_tournament(model, df)

        champion_counts[champion] += 1

        for team in knockout_teams:
            knockout_counts[team] += 1

        progress.progress((sim + 1) / simulations)

    champion_df = pd.DataFrame(
        [
            {"Team": team, "Champion Probability": count / simulations}
            for team, count in champion_counts.items()
        ]
    ).sort_values("Champion Probability", ascending=False)

    knockout_df = pd.DataFrame(
        [
            {"Team": team, "Knockout Probability": count / simulations}
            for team, count in knockout_counts.items()
        ]
    ).sort_values("Knockout Probability", ascending=False)

    return champion_df, knockout_df


st.set_page_config(page_title="2026 World Cup Predictor", layout="wide")

st.title("2026 World Cup Predictor")

model = load_model()
df = load_data()

page = st.sidebar.selectbox(
    "Choose Page",
    ["Match Predictor", "Tournament Simulator", "Groups"],
)

teams = sorted(set(df["home_team"]) | set(df["away_team"]))

if page == "Match Predictor":
    st.header("Match Predictor")

    col1, col2 = st.columns(2)

    with col1:
        team_a = st.selectbox(
            "Team A",
            teams,
            index=teams.index("Argentina") if "Argentina" in teams else 0,
        )

    with col2:
        team_b = st.selectbox(
            "Team B",
            teams,
            index=teams.index("France") if "France" in teams else 1,
        )

    if team_a == team_b:
        st.warning("Pick two different teams.")
    else:
        if st.button("Predict Match"):
            probs = predict_probs(model, df, team_a, team_b)

            st.subheader(f"{team_a} vs {team_b}")

            c1, c2, c3 = st.columns(3)

            c1.metric(f"{team_a} win", f"{probs['team_a_win']:.1%}")
            c2.metric("Draw", f"{probs['draw']:.1%}")
            c3.metric(f"{team_b} win", f"{probs['team_b_win']:.1%}")

            chart_df = pd.DataFrame(
                {
                    "Probability": [
                        probs["team_a_win"],
                        probs["draw"],
                        probs["team_b_win"],
                    ]
                },
                index=[f"{team_a} win", "Draw", f"{team_b} win"],
            )

            st.bar_chart(chart_df)

elif page == "Tournament Simulator":
    st.header("Tournament Simulator")

    simulations = st.slider(
        "Number of simulations",
        min_value=100,
        max_value=5000,
        value=1000,
        step=100,
    )

    if st.button("Run Tournament Simulation"):
        champion_df, knockout_df = run_tournament_simulations(
            model,
            df,
            simulations,
        )

        st.subheader("Champion Odds")
        st.dataframe(champion_df, use_container_width=True)

        st.bar_chart(champion_df.set_index("Team").head(15))

        st.subheader("Knockout Qualification Odds")
        st.dataframe(knockout_df, use_container_width=True)

elif page == "Groups":
    st.header("2026 Groups")

    for group_name, group_teams in GROUPS.items():
        st.subheader(f"Group {group_name}")
        st.write(", ".join(group_teams))