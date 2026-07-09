import pandas as pd

RESULTS_PATH = "data/raw/results.csv"
ELO_PATH = "data/raw/eloratings.csv"
FORMER_NAMES_PATH = "data/raw/former_names.csv"
OUTPUT_PATH = "data/processed/matches_with_elo.csv"


def load_data():
    results = pd.read_csv(RESULTS_PATH)
    elo = pd.read_csv(ELO_PATH)
    former_names = pd.read_csv(FORMER_NAMES_PATH)

    results["date"] = pd.to_datetime(results["date"], format="mixed")
    elo["date"] = pd.to_datetime(elo["date"], format="mixed")

    return results, elo, former_names


def clean_team_names(results, elo, former_names):
    name_map = dict(zip(former_names["former"], former_names["current"]))

    results["home_team"] = results["home_team"].replace(name_map)
    results["away_team"] = results["away_team"].replace(name_map)
    elo["team"] = elo["team"].replace(name_map)

    return results, elo


def add_pre_match_elo(elo):
    elo["elo_pre"] = elo["rating"] - elo["change"]
    return elo[["date", "team", "elo_pre"]]


def merge_elo(results, elo):
    results = results.sort_values("date").reset_index(drop=True)
    elo = elo.sort_values("date").reset_index(drop=True)

    # Home Elo: latest Elo for home_team on or before match date
    home_elo = elo.rename(columns={
        "team": "home_team",
        "elo_pre": "home_elo_pre"
    })

    df = pd.merge_asof(
        results,
        home_elo,
        on="date",
        by="home_team",
        direction="backward"
    )

    # Away Elo: latest Elo for away_team on or before match date
    away_elo = elo.rename(columns={
        "team": "away_team",
        "elo_pre": "away_elo_pre"
    })

    df = pd.merge_asof(
        df,
        away_elo,
        on="date",
        by="away_team",
        direction="backward"
    )

    df["elo_diff"] = df["home_elo_pre"] - df["away_elo_pre"]

    return df


def create_target(df):
    def result(row):
        if row["home_score"] > row["away_score"]:
            return 2   # home win
        elif row["home_score"] == row["away_score"]:
            return 1   # draw
        else:
            return 0   # away win

    df["target"] = df.apply(result, axis=1)
    return df


def main():
    print("Loading data...")
    results, elo, former_names = load_data()
    print("Loaded data")

    print("Cleaning team names...")
    results, elo = clean_team_names(results, elo, former_names)
    print("Cleaned team names")

    print("Adding pre-match Elo...")
    elo = add_pre_match_elo(elo)
    print("Added pre-match Elo")

    print("Merging Elo into results...")
    df = merge_elo(results, elo)
    print("Merged Elo")

    print("Creating target column...")
    df = create_target(df)
    print("Created target")

    print("Rows before dropping missing Elo:", len(df))
    print("Missing home Elo:", df["home_elo_pre"].isna().sum())
    print("Missing away Elo:", df["away_elo_pre"].isna().sum())

    missing = df[df["home_elo_pre"].isna() | df["away_elo_pre"].isna()]
    print("First 20 rows with missing Elo:")
    print(missing[["date", "home_team", "away_team"]].head(20))

    print("Dropping rows with missing Elo...")
    df = df.dropna(subset=["home_elo_pre", "away_elo_pre"])
    print("Dropped missing Elo rows")

    print("Saving file...")
    df.to_csv(OUTPUT_PATH, index=False)

    print("Saved:", OUTPUT_PATH)
    print(df.head())
    print("Rows after dropping missing Elo:", len(df))
    print("Missing home Elo after drop:", df["home_elo_pre"].isna().sum())
    print("Missing away Elo after drop:", df["away_elo_pre"].isna().sum())


if __name__ == "__main__":
    main()