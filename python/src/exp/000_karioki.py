


# %%
# テストファイルの読み込み
# =================================================
df_test = load_data(1)

# %%

# 特徴量作成4つ
df_test = data_pre01(df_test)

# 特徴量作成8つ
df_test = data_pre02(df_test)


# %%
# 推論データのデータセット作成
set_file = df_test
x_test = set_file
id_test = pd.DataFrame(set_file.index)

print(x_test.shape, id_test.shape)
# %%
x_test.head()
# %%
# カテゴリ変数をcategory型に変換
x_test = data_pre00(x_test)

# 標準化=>PCA処理
x_test = data_pre03(x_test)


# %%
# list(shap_sort["col"][:10])
x_test = x_test[
    ["T_Bil", "pc01", "AST_GOT", "AG_ratio", "ALP", "Alb/ALT_ex3", "TP", "D_Bil"]
]
x_test.shape

# %%
# 推論処理 tanntai.ver
# =================================================
test_pred = predict_lgb(
    x_test,
    id_test,
    list_nfold=[0, 1, 2, 3, 4],
)
# %%
print(test_pred.shape)
test_pred.head()
# %%
# submitファイルの出力
# =================================================


if not skip_run:
    test_pred.to_csv(
        os.path.join(OUTPUT_EXP, f"submission_{name}.csv"), index=False, header=False
    )













# %%


def drop_features(train, target):
    # 相関係数の計算
    train_ = train.drop([target], axis=1)
    corr_matrix_ = train_.corr().abs()
    corr_matrix = train.corr().abs()

    # 相関係数が0.85以上の変数ペアを抽出
    high_corr_vars = np.where(np.triu(corr_matrix_, k=1) > 0.85)
    high_corr_pairs = [
        (train_.columns[x], train_.columns[y]) for x, y in zip(*high_corr_vars)
    ]

    # 目的変数との相関係数が小さい方の変数を削除
    for pair in high_corr_pairs:
        var1_corr = corr_matrix.loc[target, pair[0]]
        var2_corr = corr_matrix.loc[target, pair[1]]

        try:  # 既に消した変数が入ってきたとき用
            if var1_corr < var2_corr:
                train = train.drop(pair[0], axis=1)
            else:
                train = train.drop(pair[1], axis=1)
        except:
            pass
    return train


train_2 = drop_features(train_, "disease")
print(train_2.shape)

# %%
train_2.info()

# %%
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr


# %%
def remove_collinear_features(train, target, threshold=1.0):
    X = train.drop(target, axis=1)
    y = train[target]
    cols = X.columns
    # 特徴量間の非類似性距離行列を計算
    std_ = StandardScaler().fit_transform(X)
    X_ = pd.DataFrame(std_, columns=X.columns)  # 標準化

    distances = np.zeros((X_.shape[1], X_.shape[1]))

    for i in range(X_.shape[1]):
        for j in range(i + 1, X_.shape[1]):
            corr, _ = spearmanr(X_.iloc[:, i], X_.iloc[:, j])
            distances[i, j] = distances[j, i] = 1 - abs(corr)
    np.fill_diagonal(distances, 0)  # 対角成分をゼロに設定
    distances = squareform(distances)

    # Ward の最小分散基準で階層的クラスター分析
    clusters = linkage(distances, method="ward")
    cluster_labels = fcluster(clusters, threshold, criterion="distance")
    # クラスター内で1つの特徴量のみ残す
    unique_cluster_labels = np.unique(cluster_labels)
    unique_features = []
    for label in unique_cluster_labels:
        features = X.columns[cluster_labels == label]
        print(f"同じクラスタの特徴量は{features}です。")
        if len(features) > 1:
            print(f"選ばれたのは{features[0]}でした。")
            unique_features.append(features[0])
        else:
            print(f"選ばれたのは{features}でした。")
            unique_features.extend(features)

    df = X[unique_features]
    df[target] = y

    return df, clusters, cols


train_3, clusters, columns = remove_collinear_features(train_, "disease", threshold=1.0)
print(train_3.shape)
print(train_3.info())


# %%

display(train_3.corr())

sns.heatmap(train_3.corr(), vmax=1, vmin=-1, annot=True)

# %%
from sklearn.feature_selection import RFECV
from sklearn.ensemble import GradientBoostingRegressor


def select_features_by_REF(train, target, n_features):
    X = train.drop(target, axis=1)
    y = train[target]
    cv = KFold(10)
    rfe = RFECV(
        estimator=GradientBoostingRegressor(random_state=0),
        min_features_to_select=n_features,
        step=0.5,
        cv=cv,
        scoring="neg_mean_absolute_error",
    )
    rfe.fit(X, y)
    train_selected = pd.DataFrame(
        columns=X.columns.values[rfe.support_],
        data=train[X.columns.values[rfe.support_]],
    )  # data=rfe.transform(X))
    train_selected[target] = y
    return train_selected


train_4 = select_features_by_REF(train_3, "disease", n_features=5)
print(train_4.shape)
# %%
train_4.info()
