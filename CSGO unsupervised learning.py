import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

game_economics = pd.read_csv("game_economics.csv")
match_viewership = pd.read_csv("match_viewership.csv")
player_stats = pd.read_csv("player_stats.csv")
team_rankings = pd.read_csv("team_rankings.csv")
tournament_results = pd.read_csv("tournament_results.csv")

role_dummies = player_stats["primary_role"].str.get_dummies(sep="/")
player_stats = pd.concat([player_stats, role_dummies], axis=1)

features = ["peak_hltv_rating", "career_kd_ratio", "career_headshot_pct"] + role_dummies.columns.tolist() #feature variables in the player_stats
X = StandardScaler().fit_transform(player_stats[features])