import pandas as pd
import numpy as np

# Data configuration
GAMES_PATH = "data/Games.csv"
TEAM_STATS_PATH = "data/TeamStatistics.csv"
PLAYERS_PATH = "data/Players.csv"
GAME_FEATURE_COLS_RAW = ["gameId", "gameDate", "hometeamId", "awayteamId", "result"]
# MODEL_FEATURE_COLS = ["avg_phys", "avg_magic", "red", "green", "blue"]
# CF_FEATURE_COLS = ["adv_id", "potion_id"]  # For collaborative filtering models
# HYBRID_FEATURE_COLS = ["adv_id", "potion_id", "avg_phys", "avg_magic", "red", "green", "blue"]  # For hybrid models
# TARGET_COL = "enjoyment"
RANDOM_SEED = 42

FEATURE_PATH = "data/feature_table.csv"
N_GAMES = 10

# cutoff date for train/test split
# (this is the first day of the 2025 season)
TRAIN_CUTOFF = "2025-10-02T12:00:00Z"

class RecommenderDataPrep:
    """Utility class for preparing recommender system data."""

    def __init__(self):
        self.games_path = GAMES_PATH
        self.team_stats_path = TEAM_STATS_PATH
        self.players_path = PLAYERS_PATH
        self.feature_cols = GAME_FEATURE_COLS_RAW
        #self.target_col = TARGET_COL
        self.random_seed = RANDOM_SEED

        self.games = None
        self.train_df = None
        self.eval_users = None
        self.candidates_by_adv = None
        self.relevant_by_adv = None
        self.adv_features = None
        self.potion_features = None
        self.adv_info = None

    def load_and_prepare(self, create_csv = True):
        """Load data and prepare train/test splits."""
        
        if create_csv:
            print("Creating feature table...")
            
            games = pd.read_csv(self.games_path)
            games = games[games["gameType"] != "Playoffs"]
            games["result"] = (games["hometeamId"] == games["winner"]).astype(int)

            games = games[GAME_FEATURE_COLS_RAW]

            team_stats = pd.read_csv(self.team_stats_path)

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

        self.train_df = self.df[self.df["gameDate"] < TRAIN_CUTOFF].copy()
        self.test_df = self.df[self.df["gameDate"] >= TRAIN_CUTOFF].copy()

        
        
        # # Per-user holdout of positives
        # train_rows = []
        # heldout_positives = {}

        # for aid, g in self.games.groupby("adv_id"):
        #     likes = g[g["liked"] == 1]
        #     if len(likes) >= 1:
        #         n_hold = min(self.holdout_per_user, len(likes))
        #         test_likes = likes.sample(n=n_hold, random_state=self.random_seed)
        #         heldout_positives[aid] = set(test_likes["potion_id"].tolist())
        #         train_rows.append(g.drop(test_likes.index))
        #     else:
        #         train_rows.append(g)

        # self.train_df = pd.concat(train_rows, ignore_index=True)

        # # Candidate sets
        # self.eval_users = sorted(heldout_positives.keys())
        # all_potions = sorted(self.games["potion_id"].unique().tolist())
        # seen_train_by_adv = {aid: set(g["potion_id"].tolist())
        #                     for aid, g in self.train_df.groupby("adv_id")}
        # self.candidates_by_adv = {
        #     aid: [pid for pid in all_potions if pid not in seen_train_by_adv.get(aid, set())]
        #     for aid in self.eval_users
        # }

        # self.relevant_by_adv = {
        #     aid: (heldout_positives[aid] & set(self.candidates_by_adv.get(aid, [])))
        #     for aid in self.eval_users
        # }

        # # Feature lookups
        # self.adv_features = self.train_df.groupby("adv_id")[["avg_phys", "avg_magic"]].first()
        # self.potion_features = self.games.groupby("potion_id")[["red", "green", "blue"]].first()
        # self.adv_info = self.games.groupby("adv_id")[["class", "level"]].first()

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
        y_train = self.train_df[self.target_col].astype(float)
        return X_train, y_train

    def _create_cf_interactions(self, aid, candidates):
        """Create feature matrix for collaborative filtering (IDs only)."""
        n = len(candidates)
        X_cand = pd.DataFrame({
            "adv_id": [aid] * n,
            "potion_id": candidates
        })
        return X_cand, candidates

    def _create_content_interactions(self, candidates, adv_phys, adv_magic):
        """Create feature matrix for content-based models (features only)."""
        pot_feats = self.potion_features.loc[candidates][["red", "green", "blue"]].values.astype(float)
        n = len(candidates)

        X_cand = pd.DataFrame(
            np.column_stack([
                np.full(n, float(adv_phys)),
                np.full(n, float(adv_magic)),
                pot_feats
            ]),
            columns=["avg_phys", "avg_magic", "red", "green", "blue"]
        )
        return X_cand, candidates

    def _create_hybrid_interactions(self, aid, candidates, adv_phys, adv_magic):
        """Create feature matrix for hybrid models (IDs + features)."""
        pot_feats = self.potion_features.loc[candidates][["red", "green", "blue"]].values.astype(float)
        n = len(candidates)

        X_cand = pd.DataFrame(
            np.column_stack([
                np.full(n, float(aid)),
                np.array(candidates, dtype=float),
                np.full(n, float(adv_phys)),
                np.full(n, float(adv_magic)),
                pot_feats
            ]),
            columns=["adv_id", "potion_id", "avg_phys", "avg_magic", "red", "green", "blue"]
        )
        return X_cand, candidates

    def create_unseen_interactions(self, aid, model_feature_cols: list):
        """Create feature matrix for unseen interactions for a given adventurer."""
        candidates = self.candidates_by_adv[aid]

        # Check if this is a CF model (only needs adv_id and potion_id)
        if set(model_feature_cols) == {"adv_id", "potion_id"}:
            return self._create_cf_interactions(aid, candidates)

        # For hybrid or content-based models: need features
        adv_phys, adv_magic = self.adv_features.loc[aid][["avg_phys", "avg_magic"]]

        # Check if hybrid model (includes IDs + features)
        if "adv_id" in model_feature_cols and "potion_id" in model_feature_cols:
            return self._create_hybrid_interactions(aid, candidates, adv_phys, adv_magic)
        else:
            # Content-based only
            return self._create_content_interactions(candidates, adv_phys, adv_magic)

    def get_positive_rate(self):
        """Calculate global positive rate."""
        return self.games["liked"].mean()

    def get_user_count(self):
        """Get total number of unique users."""
        return self.games["adv_id"].nunique()