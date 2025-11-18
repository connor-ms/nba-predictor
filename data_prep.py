import pandas as pd
import numpy as np
import kagglehub

# Data configuration
GAME_FEATURE_COLS_RAW = ["gameId", "gameDate", "hometeamId", "awayteamId", "result"]
MODEL_FEATURE_COLS = ["home_efg", "away_efg", "home_tov", "away_tov", "home_orb", "away_orb", "home_ft", "away_ft"]
TARGET_COL = "result"
RANDOM_SEED = 42

FEATURE_PATH = "data/feature_table.csv"
N_GAMES = 10

# cutoff date for train/test split
# (this is the first day of the 2025 season)
TRAIN_CUTOFF = "2025-10-02T12:00:00Z"

class RecommenderDataPrep:
    """Utility class for preparing recommender system data."""

    def __init__(self):
        self.feature_cols = GAME_FEATURE_COLS_RAW
        self.target_col = TARGET_COL
        self.random_seed = RANDOM_SEED

        self.train_df = None
        self.test_df = None

    def load_and_prepare(self, create_csv = True):
        """Load data and prepare train/test splits."""

        data_path = kagglehub.dataset_download("eoinamoore/historical-nba-data-and-player-box-scores")
        
        if create_csv:
            print("Creating feature table...")
            
            games = pd.read_csv(data_path + "/Games.csv")
            games = games[games["gameType"] != "Playoffs"]
            games["result"] = (games["hometeamId"] == games["winner"]).astype(int)

            games = games[GAME_FEATURE_COLS_RAW]

            team_stats = pd.read_csv(data_path + "/TeamStatistics.csv")

            team_stats_sorted = team_stats.sort_values(by="gameDate", ascending=False)

            home_stats = team_stats_sorted[team_stats_sorted["home"] == 1].copy()
            away_stats = team_stats_sorted[team_stats_sorted["home"] == 0].copy()

            opp_home_stats = away_stats[["gameId", "reboundsDefensive", "reboundsOffensive"]].rename(
                columns={"reboundsDefensive": "reboundsDefensive_opp", "reboundsOffensive": "reboundsOffensive_opp"}
            )
            opp_away_stats = home_stats[["gameId", "reboundsDefensive", "reboundsOffensive"]].rename(
                columns={"reboundsDefensive": "reboundsDefensive_opp", "reboundsOffensive": "reboundsOffensive_opp"}
            )

            home_stats = home_stats.merge(opp_home_stats, on="gameId", how="left")
            away_stats = away_stats.merge(opp_away_stats, on="gameId", how="left")

            home_agg = self.get_rolling_stats(games, home_stats, True)
            away_agg = self.get_rolling_stats(games, away_stats, False)

            games["home_efg"] = (home_agg["fieldGoalsMade"] + 0.5 * home_agg["threePointersMade"]) / home_agg["fieldGoalsAttempted"]
            games["away_efg"] = (away_agg["fieldGoalsMade"] + 0.5 * away_agg["threePointersMade"]) / away_agg["fieldGoalsAttempted"]

            games["home_tov"] = home_agg["turnovers"] / (home_agg["fieldGoalsAttempted"] + 0.44 * home_agg["freeThrowsAttempted"] + home_agg["turnovers"])
            games["away_tov"] = away_agg["turnovers"] / (away_agg["fieldGoalsAttempted"] + 0.44 * away_agg["freeThrowsAttempted"] + away_agg["turnovers"])

            games["home_orb"] = home_agg["reboundsOffensive"] / (home_agg["reboundsOffensive"] + home_agg["reboundsDefensive_opp"])
            games["away_orb"] = away_agg["reboundsOffensive"] / (away_agg["reboundsOffensive"] + away_agg["reboundsDefensive_opp"])

            games["home_ft"] = home_agg["freeThrowsMade"] / home_agg["freeThrowsAttempted"]
            games["away_ft"] = away_agg["freeThrowsMade"] / away_agg["freeThrowsAttempted"]

            games.to_csv(FEATURE_PATH, index=False)
            print("Wrote feature table to file")
        
        self.df = pd.read_csv(FEATURE_PATH)

        #hacky
        self.df = self.df.replace([np.inf, -np.inf], np.nan).dropna()

        self.train_df = self.df[self.df["gameDate"] < TRAIN_CUTOFF].copy()
        self.test_df = self.df[self.df["gameDate"] >= TRAIN_CUTOFF].copy()

        #largest_value = self.train_df[MODEL_FEATURE_COLS].sort_values(by="home_efg")
        #print(f"{largest_value}")

    def get_rolling_stats(self, games_df, stats_df, is_home):
        results = []
        stat_cols = ["fieldGoalsMade", "fieldGoalsAttempted", "threePointersMade", "turnovers", 
                    "freeThrowsMade", "freeThrowsAttempted", "assists", "reboundsDefensive", 
                    "reboundsOffensive", "reboundsDefensive_opp", "reboundsOffensive_opp"]
        
        team_col = "hometeamId" if is_home else "awayteamId"
        
        for _, game in games_df.iterrows():
            team_id = game[team_col]
            game_date = game["gameDate"]
            game_id = game["gameId"]
            
            mask = (
                (stats_df["teamId"] == team_id) &
                (stats_df["gameDate"] <= game_date) &
                (stats_df["gameId"] != game_id)
            )
            
            team_games = stats_df[mask].head(N_GAMES)
            
            if len(team_games) > 0:
                agg = team_games[stat_cols].sum() / N_GAMES
            else:
                agg = pd.Series({col: 0 for col in stat_cols})
            
            results.append(agg)
        
        return pd.DataFrame(results, index=games_df.index)

    def get_training_data(self, model_feature_cols: list):
        """Get training features and target."""
        X_train = self.train_df[model_feature_cols].astype(float)
        y_train = self.train_df[self.target_col].astype(int)
        return X_train, y_train
    
    def get_test_data(self, model_feature_cols):
        X_test = self.test_df[model_feature_cols]
        return X_test
    
    def get_test_results(self):
        return self.test_df[self.target_col]

    def get_positive_rate(self):
        """Calculate global positive rate."""
        return self.df[TARGET_COL].mean()

    def get_user_count(self):
        """Get total number of unique users."""
        #return self.df["adv_id"].nunique()