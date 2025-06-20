def preprocess_grouped_data(df_country, round_nr, binary, categorical, ordered_categorical, numerical, binary_neg, binary_pos):
    df_round = df_country[df_country['Round'] == round_nr].copy()

    round_binary = [col for col in binary if col in df_round.columns]
    round_categorical = [col for col in categorical if col in df_round.columns]
    round_ordered = [col for col in ordered_categorical if col in df_round.columns]
    round_numerical = [col for col in numerical if col in df_round.columns]

    df_round = df_round[round_binary + round_categorical + round_ordered + round_numerical + ['Groupnr']].copy()

    for col in round_binary:
        df_round[col] = pd.to_numeric(df_round[col], errors='coerce')
    for col in binary_neg:
        if col in df_round:
            df_round.loc[df_round[col] == 2.0, col] = 1.0
            df_round.loc[df_round[col] == 1.0, col] = 0.0
    for col in binary_pos:
        if col in df_round:
            df_round.loc[df_round[col] == 1.0, col] = 1.0
            df_round.loc[df_round[col] == 2.0, col] = 0.0
    for col in round_binary:
        df_round.loc[~df_round[col].isin([0.0, 1.0]) & df_round[col].notna(), col] = np.nan

    for col in round_categorical:
        df_round[col] = df_round[col].astype(str).fillna('nan')
    cat_df = pd.DataFrame()
    if round_categorical:
        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        cat_encoded = ohe.fit_transform(df_round[round_categorical])
        cat_df = pd.DataFrame(cat_encoded, columns=ohe.get_feature_names_out(), index=df_round.index)

    df_features = df_round[round_binary + round_ordered + round_numerical]
    if not cat_df.empty:
        df_features = pd.concat([df_features, cat_df], axis=1)

    df_grouped = df_features.groupby(df_round['Groupnr']).mean()
    df_grouped.fillna(df_grouped.mean(skipna=True), inplace=True)
    df_grouped.fillna(0, inplace=True)

    return df_grouped

def plot_clusters(X, labels, method_title):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    plt.figure(figsize=(6, 5))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette="tab10")
    plt.title(method_title)
    plt.xlabel(f"PCA 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    plt.ylabel(f"PCA 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.grid(True)
    plt.show()



def plot_silhouette_scores(scores_dict, title):
    plt.figure(figsize=(6, 4))
    ks = list(scores_dict.keys())
    scores = list(scores_dict.values())
    plt.plot(ks, scores, marker='o')
    plt.xticks(ks)
    plt.title(f"Silhouette Scores - {title}")
    plt.xlabel("Aantal clusters")
    plt.ylabel("Silhouette score")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_dendrogram(Z, method_title):
    plt.figure(figsize=(12, 6))
    dendrogram(Z, no_labels=True)
    plt.title(method_title)
    plt.xlabel("Groups")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.show()

def find_best_k(X_scaled, max_k=4, plot_title=None):
    best_k, best_score = 2, -1
    silhouette_scores = {}
    for k in range(2, min(max_k + 1, len(X_scaled))):
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = kmeans.fit_predict(X_scaled)
        try:
            score = silhouette_score(X_scaled, labels)
            silhouette_scores[k] = score
            if score > best_score:
                best_score = score
                best_k = k
        except:
            continue
    if plot_title:
        plot_silhouette_scores(silhouette_scores, plot_title)
    return best_k

def find_best_hc_k(X_scaled, max_k=4, plot_title=None):
    Z = linkage(X_scaled, method='ward')
    best_k, best_score, best_labels = 2, -1, None
    silhouette_scores = {}
    for k in range(2, min(max_k + 1, len(X_scaled))):
        labels = fcluster(Z, t=k, criterion='maxclust')
        try:
            score = silhouette_score(X_scaled, labels)
            silhouette_scores[k] = score
            if score > best_score:
                best_score = score
                best_k = k
                best_labels = labels
        except:
            continue
    if plot_title:
        plot_silhouette_scores(silhouette_scores, plot_title)
    return best_k, best_labels

def find_best_dbscan(X_scaled, eps_values=[5, 10, 15, 20, 25], min_samples_values=[2, 3, 4]):
    best_score, best_params, best_labels = -1, {"eps": None, "min_samples": None}, None
    for eps in eps_values:
        for min_samples in min_samples_values:
            db = DBSCAN(eps=eps, min_samples=min_samples)
            labels = db.fit_predict(X_scaled)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            if n_clusters < 2:
                continue
            try:
                score = silhouette_score(X_scaled, labels)
                if score > best_score:
                    best_score = score
                    best_params = {"eps": eps, "min_samples": min_samples}
                    best_labels = labels
            except:
                continue
    return best_params, best_labels

def apply_clustering_algorithms(df_grouped, country_code, round_nr, max_k=5):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_grouped)

    best_k = find_best_k(X_scaled, max_k=4, plot_title=f"{country_code} R{round_nr} - KMeans") if X_scaled.shape[0] >= 4 else 2
    kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=42)
    kmeans_labels = kmeans.fit_predict(X_scaled)
    plot_clusters(X_scaled, kmeans_labels, f"{country_code} R{round_nr} - KMeans (k={best_k})")

    Z = linkage(X_scaled, method='ward')
    plot_dendrogram(Z, f"{country_code} R{round_nr} - Hierarchical Dendrogram")

    hc_best_k, hc_labels = find_best_hc_k(X_scaled, max_k=4, plot_title=f"{country_code} R{round_nr} - Hierarchical")
    hc_labels = hc_labels - 1
    plot_clusters(X_scaled, hc_labels, f"{country_code} R{round_nr} - Hierarchical (k={hc_best_k})")

    dbscan_params, dbscan_labels = find_best_dbscan(X_scaled)
    if dbscan_labels is not None:
        dbscan_labels = np.array([
            -1 if lbl == -1 else lbl - min(dbscan_labels[dbscan_labels != -1])
            for lbl in dbscan_labels
        ])
        n_dbscan_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
        plot_clusters(X_scaled, dbscan_labels, f"{country_code} R{round_nr} - DBSCAN (k={n_dbscan_clusters})")

for country_code, df_country in {
    'GHA': df_gha,
    'RWA': df_rwa,
    'UGA': df_uga,
    'CIV': df_civ
}.items():
    rounds = df_country['Round'].dropna().unique()
    rounds = sorted(rounds, key=lambda x: int(float(x)) if str(x).replace('.', '', 1).isdigit() else float('inf'))
    for round_nr in rounds:
        if str(round_nr) in ['1', '3', '101', '102']:
            continue
        df_grouped = preprocess_grouped_data(
            df_country, round_nr,
            binary=binary,
            categorical=categorical,
            ordered_categorical=ordered_categorical,
            numerical=numerical,
            binary_neg=binary_neg,
            binary_pos=binary_pos
        )
        if not df_grouped.empty and df_grouped.shape[0] >= 2:
            apply_clustering_algorithms(df_grouped, country_code, round_nr)


