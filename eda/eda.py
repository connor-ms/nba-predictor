import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from sklearn.preprocessing import StandardScaler



# Everything below is for the old dataset
games = pd.read_csv('data/games.csv')
details = pd.read_csv('data/games_details.csv')
teams = pd.read_csv('data/teams.csv')

print(f"games.csv entry count: {len(games)}")
print(f"games_details.csv entry count: {len(details)}")
print(f"teams.csv entry count: {len(teams)}")

print(games.duplicated(subset=["GAME_ID"]).sum())

missing_games = games.isnull().mean().sort_values(ascending=False)
print("\nMissing values in games.csv:")
print(missing_games[missing_games > 0])

missing_details = details.isnull().mean().sort_values(ascending=False)
print("\nMissing values in games_details.csv (top 10):")
print(missing_details.head(10))

win_rate = games['HOME_TEAM_WINS'].mean()
print(f"\nHome team win rate: {win_rate:.2%}")

team_stats = details.groupby(['GAME_ID', 'TEAM_ID']).agg({
    'PTS': 'sum',
    'REB': 'sum',
    'AST': 'sum',
    'TO': 'sum',
    'STL': 'sum',
    'BLK': 'sum',   
    'FG_PCT': 'mean',
    'FT_PCT': 'mean',
    'FG3_PCT': 'mean'
}).reset_index()

home = team_stats.merge(games[['GAME_ID','HOME_TEAM_ID']], left_on=['GAME_ID','TEAM_ID'], right_on=['GAME_ID','HOME_TEAM_ID'], how='inner')
away = team_stats.merge(games[['GAME_ID','VISITOR_TEAM_ID']], left_on=['GAME_ID','TEAM_ID'], right_on=['GAME_ID','VISITOR_TEAM_ID'], how='inner')

home = home.add_suffix('_home')
away = away.add_suffix('_away')

merged = home.merge(away, left_on='GAME_ID_home', right_on='GAME_ID_away', how='inner').rename(columns={'GAME_ID_home':'GAME_ID'})

merged = merged.merge(games[['GAME_ID','HOME_TEAM_WINS']], on='GAME_ID')
merged['label'] = merged['HOME_TEAM_WINS']

diff_cols = ['PTS', 'REB', 'AST', 'TO', 'STL', 'BLK', 'FG_PCT', 'FT_PCT', 'FG3_PCT']
for col in diff_cols:
    merged[f'{col}_diff'] = merged[f'{col}_home'] - merged[f'{col}_away']

plt.figure(figsize=(12,5))
sns.countplot(x='label', data=merged, palette='coolwarm')
plt.title('Home Team Win vs Loss Count')
plt.xlabel('Home Team Win (1) or Loss (0)')
plt.ylabel('Games')
plt.show()

plt.figure(figsize=(10,5))
sns.histplot(merged['PTS_diff'], bins=40, kde=True)
plt.title('Distribution of Points Difference (Home - Away)')
plt.xlabel('Points Difference')
plt.show()

plt.figure(figsize=(10,7))
corr = merged[[f'{col}_diff' for col in diff_cols] + ['label']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap of Feature Differences and Win Label')
plt.show()

plt.figure(figsize=(8,6))
sns.scatterplot(
    x='REB_diff', y='AST_diff',
    hue='label', data=merged,
    alpha=0.6, palette='coolwarm'
)
plt.title('Rebounds vs Assists Difference by Game Outcome')
plt.xlabel('Rebounds Difference (Home - Away)')
plt.ylabel('Assists Difference (Home - Away)')
plt.show()

X = merged[[f'{col}_diff' for col in diff_cols]].fillna(0)
X_scaled = StandardScaler().fit_transform(X)

reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
embedding = reducer.fit_transform(X_scaled)

plt.figure(figsize=(10,7))
sns.scatterplot(x=embedding[:,0], y=embedding[:,1], hue=merged['label'], palette='coolwarm', alpha=0.7)
plt.title('UMAP Projection of Game-Level Features')
plt.xlabel(None)
plt.ylabel(None)
plt.legend(title='Home Win')
plt.show()

