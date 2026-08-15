import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

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

#giving clusters with name
cluster_labels = {
    0: "Rifler",
    1: "AWPer",
    2: "IGL",
    3: "Outlier"
}

#pipeline
scaler = StandardScaler()
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
pipeline = make_pipeline(scaler, kmeans)
pipeline.fit(player_stats[features]) 
labels = pipeline.predict(player_stats[features])

player_stats["cluster"] = labels
player_stats["cluster_label"] = player_stats["cluster"].map(cluster_labels)

#cross tabulation
ct = pd.crosstab(player_stats["primary_role"], player_stats["cluster_label"])
ct_pct = pd.crosstab(player_stats["primary_role"], player_stats["cluster_label"], normalize="columns") * 100


#printing
print(labels)
print("")
print(player_stats.groupby("cluster")[features].mean().round(2))
print("")
print("Inertia:", round(kmeans.inertia_, 2))
print("")
print(ct)
print("")
print(ct_pct.round(2))
print("")

#cluster hierarchy
X_scaled = pipeline.named_steps["standardscaler"].transform(player_stats[features])
mergings = linkage(X_scaled, method="complete")
plt.figure(figsize=(10, 14))
dendrogram(mergings, labels=player_stats["player_handle"].values, orientation="left", leaf_font_size=6)
plt.title("Player Cluster Hierarchy")
plt.ylabel("Distance")
plt.savefig("dendrogram.png")

#inertia plot display
inertias = []
k_range = range(1, 7)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(player_stats[features])
    inertias.append(km.inertia_)

plt.figure()
plt.plot(k_range, inertias, marker="o")
plt.axvline(x=4, color="red", linestyle="--", label="k=4 (chosen)")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.legend()
plt.savefig("elbow.png")

#t-SNE transformation
tsne = TSNE(learning_rate=100)
tsne_features = tsne.fit_transform(X_scaled)

xs = tsne_features[:, 0]
ys = tsne_features[:, 1]
plt.figure()
plt.scatter(xs, ys, c=player_stats["cluster"], cmap="viridis")
plt.title("t-SNE of Player Clusters")
plt.savefig("tsne.png")
plt.show()

#scatterplot display 
scatter = plt.scatter(player_stats["career_kd_ratio"], player_stats["career_headshot_pct"], 
                       c=player_stats["cluster"], cmap="viridis")
plt.xlabel("K/D Ratio")
plt.ylabel("Headshot %")
plt.title("Player Clusters")
plt.legend(*scatter.legend_elements(), title="Cluster")

plt.show()