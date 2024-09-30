


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







# %% ファイルの読み込み
# Load Data
# =================================================
# test

df_test = load_data(2)
display(df_test.shape)
df_test.info()


# %%
df_test.columns
# %%
# これまでの処理
# ==========================================================
# 特徴量作成4つ
df_test = data_ft01(df_test)
# 特徴量作成8つ
df_test = data_ft02(df_test)


#

# %%
set_file = df_test
x_test01 = set_file.drop([target_columns], axis=1)
id_test01 = pd.DataFrame(set_file.index)

print(x_test01.shape,, id_test01.shape)

x_test01 = data_pre_catg(x_test01)
# 標準化=>PCA処理
# Genderを数値化
x_test01["Gender"] = pd.get_dummies(
    x_test01["Gender"], drop_first=True, dtype="uint8"
)
x_test01 = data_pca(x_test01, 8)
x_test01.columns

# %%

x_test01 = x_test01[
    ["T_Bil", "pc00", "AST_GOT", "AG_ratio", "ALP", "Alb/ALT_ex3", "TP", "D_Bil"]
]

x_test01.shape

# %%学習モデル01 lgbm

# =================================================
# =================================================
# =================================================


# %% [markdown]
## Main 分析start!
# =========================================================

# %%
df_test = load_data(2)
display(df_test.shape)
df_test.info()
# Genderを数値化
df_test["Gender"] = pd.get_dummies(df_test["Gender"], drop_first=True, dtype="uint8")
# %%特徴量作成
df_test66 = df_test.copy()
features = df_test66.drop(["disease", "Gender", "Age"], axis=1).columns
print("特徴量生成前：", df_test66.shape)
test_ = create_features(df_test66, features, ["disease", "Gender", "Age"])
print("特徴量生成後：", test_.shape)

print(test_.info)
# %%
test_.columns


# %%
# 相関高いもの消す
test_2 = drop_features(test_, "disease")
print(test_2.shape)
# %%
test_2.columns


# %%
# クラス分け一つ目
# remove_collinear_features1 前から1つ
test_3_1, clusters, columns = remove_collinear_features(
    test_2, "disease", threshold=1.0, s=0
)
print(test_3_1.shape)
print(test_3_1.columns)
# %%
# 元のやつと結合して、重複削除
df_test_3_1 = pd.concat([df_test, test_3_1], axis=1)
df_test_3_1 = df_test_3_1.loc[:, ~df_test_3_1.columns.duplicated()]
df_test_3_1.columns

# %%
# データセット作成
# =================================================
set_file = df_test_3_1
x_test_3_1 = set_file.drop([target_columns], axis=1)
id_test_3_1 = pd.DataFrame(set_file.index)


print(x_test_3_1.shape,  id_test_3_1.shape)

#
# %%コラムセレクト
# なし


# %%学習モデル02 lgbm

# =================================================
# =================================================
# =================================================
# %%
# クラス分け2つ目
# remove_collinear_features2 後ろから１
test_3_2, clusters, columns = remove_collinear_features(
    test_2, "disease", threshold=1.0, s=-1
)
print(test_3_2.shape)
print(test_3_2.columns)

# %%
# 元のやつと結合して、重複削除
df_test_3_2 = pd.concat([df_test, test_3_2], axis=1)
df_test_3_2 = df_test_3_2.loc[:, ~df_test_3_2.columns.duplicated()]
df_test_3_2.columns

# %%
# データセット作成
# =================================================
set_file = df_test_3_2
x_test_3_2 = set_file.drop([target_columns], axis=1)
id_test_3_2 = pd.DataFrame(set_file.index)

print(x_test_3_2.shape,  id_test_3_2.shape)
# %%
# 標準化=>PCA処理
x_test_3_2 = data_pca(x_test_3_2, 8)
#%%
x_test_3_2.head()
# %%
# ベストセレクション
list2 = [
    "T_Bil",
    "pc00",
    "ALT_GPT",
    "AG_ratio/D_Bil",
    "AST_GOT",
    "AG_ratio",
    "ALP",
    "AG_ratio/AST_GOT",
    "TP",
]

x_test_3_2 = x_test_3_2[list2]

# %%学習モデル03 lgbm

# =================================================
# =================================================
# =================================================

# %%
# クラス分け3つ目
# remove_collinear_features3 後ろから2
test_3_3, clusters, columns = remove_collinear_features(
    test_2, "disease", threshold=1.0, s=-2
)
print(test_3_3.shape)
print(test_3_3.columns)

# %%
# 元のやつと結合して、重複削除
df_test_3_3 = pd.concat([df_test, test_3_3], axis=1)
df_test_3_3 = df_test_3_3.loc[:, ~df_test_3_3.columns.duplicated()]
df_test_3_3.columns

# %%
# データセット作成
# =================================================
set_file = df_test_3_3
x_test_3_3 = set_file.drop([target_columns], axis=1)
id_test_3_3 = pd.DataFrame(set_file.index)

print(x_test_3_3.shape,  id_test_3_3.shape)
# %%
# 標準化=>PCA処理
x_test_3_3 = data_pca(x_test_3_3, 8)
#%%
x_test_3_3.head()
# %%
# ベストセレクション
list3 =[
    'T_Bil',
    'pc00',
    'AG_ratio/ALT_GPT',
    'AST_GOT',
    'AG_ratio',
    'ALP',
    'TP',
    'D_Bil',
    'Alb/T_Bil',
    'TP/ALP']


x_test_3_3 = x_test_3_3[list3]

# %%学習モデル04 lgbm

# =================================================
# =================================================
# =================================================




#%%
# %%
# クラス分け4つ目
# remove_collinear_features3 後ろから2
test_3_4, clusters, columns = remove_collinear_features(
    test_2, "disease", threshold=1.0, s=1
)
print(test_3_4.shape)
print(test_3_4.columns)

# %%
# 元のやつと結合して、重複削除
df_test_3_4 = pd.concat([df_test, test_3_4], axis=1)
df_test_3_4 = df_test_3_4.loc[:, ~df_test_3_4.columns.duplicated()]
df_test_3_4.columns

# %%
# データセット作成
# =================================================
set_file = df_test_3_4
x_test_3_4 = set_file.drop([target_columns], axis=1)
id_test_3_4 = pd.DataFrame(set_file.index)

print(x_test_3_4.shape,  id_test_3_4.shape)
# %%
# 標準化=>PCA処理
# x_test_3_4 = data_pca(x_test_3_4, 8)
#%%
x_test_3_4.head()
# %%
# ベストセレクション
# なし
list4 =['T_Bil',
    'ALT_GPT',
    'T_Bil/AG_ratio',
    'AST_GOT',
    'AG_ratio',
    'ALP',
    'D_Bil',
    'TP',
    'ALP/AST_GOT',
    'D_Bil/T_Bil']

x_test_3_4 = x_test_3_4[list4]

# %%学習モデル05 lgbm

# =================================================
# =================================================
# =================================================
