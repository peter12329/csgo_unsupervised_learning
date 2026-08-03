import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

game_economics = pd.read_csv("data sets/game_economics.csv")
match_viewership = pd.read_csv("data sets/match_viewership.csv")
player_stats = pd.read_csv("data sets/player_stats.csv")
team_rankings = pd.read_csv("data sets/team_rankings.csv")
tournament_results = pd.read_csv("data sets/tournament_results.csv")

role_dummies = player_stats["primary_role"].str.get_dummies(sep="/")
player_stats = pd.concat([player_stats, role_dummies], axis=1)

features = ["peak_hltv_rating", "career_kd_ratio", "career_headshot_pct"] + role_dummies.columns.tolist() #feature variables in the player_stats
X = StandardScaler().fit_transform(player_stats[features])

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)

kmeans.fit(X)
labels = kmeans.predict(X)

print(labels)

player_stats["cluster"] = labels
print(player_stats[["player_handle", "cluster"]].sort_values("cluster", ascending=False))

scatter = plt.scatter(player_stats["career_kd_ratio"], player_stats["career_headshot_pct"], 
                       c=player_stats["cluster"], cmap="viridis")
plt.xlabel("K/D Ratio")
plt.ylabel("Headshot %")
plt.title("Player Clusters")
plt.legend(*scatter.legend_elements(), title="Cluster")
plt.show()