import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

#data sets
game_economics = pd.read_csv("data sets/game_economics.csv")
match_viewership = pd.read_csv("data sets/match_viewership.csv")
player_stats = pd.read_csv("data sets/player_stats.csv")
team_rankings = pd.read_csv("data sets/team_rankings.csv")
tournament_results = pd.read_csv("data sets/tournament_results.csv")

#dummy variables, i havent used?
role_dummies = player_stats["primary_role"].str.get_dummies(sep="/")
player_stats = pd.concat([player_stats, role_dummies], axis=1)

#feature variables 
features = ["peak_hltv_rating", "career_kd_ratio", "career_headshot_pct"] + role_dummies.columns.tolist() #feature variables in the player_stats

#scaler
X = StandardScaler().fit_transform(player_stats[features])

#kmeans
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans.fit(X)
labels = kmeans.predict(X)

#labels
print(labels)
print("")

#cluster display
player_stats["cluster"] = labels

#giving clusters with name
cluster_labels = {
    0: "Rifler",
    1: "AWPer",
    2: "IGL",
    3: "Outlier"
}

player_stats["cluster_label"] = player_stats["cluster"].map(cluster_labels)

#cluster labels:
print(player_stats.groupby("cluster")[features].mean().round(2))


#cross tabulation trying to find corr
ct = pd.crosstab(player_stats["primary_role"], player_stats["cluster_label"])
print(ct)
print("")
ct_pct = pd.crosstab(player_stats["primary_role"], player_stats["cluster_label"], normalize="columns") * 100
print(ct_pct.round(1))

#plt display 
scatter = plt.scatter(player_stats["career_kd_ratio"], player_stats["career_headshot_pct"], 
                       c=player_stats["cluster"], cmap="viridis")
plt.xlabel("K/D Ratio")
plt.ylabel("Headshot %")
plt.title("Player Clusters")
plt.legend(*scatter.legend_elements(), title="Cluster")
#plt.show()