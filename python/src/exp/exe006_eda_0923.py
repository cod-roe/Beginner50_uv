# %% [markdown]
## EDA！
# =================================================
# EDAとデータ加工

# %%
# ライブラリ読み込み
# =================================================
import datetime as dt

# import gc
# import json
import logging

# import re
import os
import sys
import pickle
from IPython.display import display
import warnings
# import zipfile

import numpy as np
import pandas as pd


# 可視化
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib


# lightGBM
import lightgbm as lgb

# sckit-learn
# from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder

from sklearn.model_selection import StratifiedKFold  # ,train_test_split, KFold
from sklearn.metrics import roc_auc_score  # accuracy_score,confusion_matrix


# %%
# Config
# =================================================

######################
# serial #
######################
serial_number = 6  # スプレッドシートAの番号

######################
# Data #
######################
comp_name = "Chronic_liver_disease"
# 評価：AUC（Area Under the Curve）

skip_run = False  # 飛ばす->True，飛ばさない->False

######################
# filename
######################
# vscode用
abs_path = os.path.abspath(__file__)  # /tmp/work/src/exp/_.py'
name = os.path.splitext(os.path.basename(abs_path))[0]
# Google Colab等用（取得できないためファイル名を入力）
# name = 'run001'

######################
# set dirs #
######################
DRIVE = os.path.dirname(os.getcwd())  # このファイルの親(scr)
INPUT_PATH = f"../input/{comp_name}/"  # 読み込みファイル場所
OUTPUT = os.path.join(DRIVE, "output")
OUTPUT_EXP = os.path.join(OUTPUT, name)  # 情報保存場所
EXP_MODEL = os.path.join(OUTPUT_EXP, "model")  # 学習済みモデル保存

######################
# Dataset #
######################
target_columns = "disease"
# sub_index = "index"

######################
# ハイパーパラメータの設定
######################
# lgbm初期値
params = {
    "boosting_type": "gbdt",
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 32,
    "n_estimators": 10000,
    "random_state": 123,
    "importance_type": "gain",
}


# %%
# Utilities #
# =================================================


# 今の日時
def dt_now():
    dt_now = dt.datetime.now()
    return dt_now


# stdout と stderr をリダイレクトするクラス
class StreamToLogger:
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level
        self.linebuf = ""

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.level, line.rstrip())

    def flush(self):
        pass


# %%
# make dirs
# =================================================
def make_dirs():
    for d in [EXP_MODEL]:
        os.makedirs(d, exist_ok=True)
    print("フォルダ作成完了")


# %%
# ファイルの確認
# =================================================
def file_list(input_path):
    file_list = []
    for dirname, _, _filenames in os.walk(input_path):
        for i, _datafilename in enumerate(_filenames):
            print("=" * 20)
            print(i, _datafilename)
            file_list.append([_datafilename, os.path.join(dirname, _datafilename)])
    file_list = pd.DataFrame(file_list, columns=["ファイル名", "ファイルパス"])
    display(file_list)
    return file_list


# %%
# メモリ削減関数
# =================================================
def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024**2
    print(f"Memory usage of dataframe is {start_mem:.2f} MB")

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:  # noqa: E721
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astypez(np.float64)
        else:
            pass

    end_mem = df.memory_usage().sum() / 1024**2
    print(f"Memory usage after optimization is: {end_mem:.2f} MB")
    print(f"Decreased by {100*((start_mem - end_mem) / start_mem):.2f}%")

    return df


# %%
# ファイルの読み込み
# =================================================
def load_data(file_index):
    # file_indexを引数に入力するとデータを読み込んでくれる
    if file_list["ファイル名"][file_index][-3:] == "csv":
        print(f"読み込んだファイル：{file_list['ファイル名'][file_index]}")
        df = reduce_mem_usage(pd.read_csv(file_list["ファイルパス"][file_index]))
        print(df.shape)
        display(df.head())

    elif file_list["ファイル名"][file_index][-3:] == "pkl" or "pickle":
        print(f"読み込んだファイル：{file_list['ファイル名'][file_index]}")
        df = reduce_mem_usage(pd.read_pickle(file_list["ファイルパス"][file_index]))
        print(df.shape)
        display(df.head())
    return df


# %%
# 前処理の定義 カテゴリ変数をcategory型に
# =================================================
def data_pre00(df):
    for col in df.columns:
        if df[col].dtype == "O":
            df[col] = df[col].astype("category")
    print("カテゴリ変数をcategory型に変換しました")
    df.info()
    return df


# %%
# 特徴量生成,one_hot exp002
# =================================================
def data_pre01(df):
    # Genderを数値化
    df["Gender"] = pd.get_dummies(df_train["Gender"], drop_first=True, dtype="uint8")
    # 特徴量1:直接ビリルビン/総ビリルビン比（D/T比） 3/2
    df["D_div_T_ex2"] = df["D_Bil"] / df["T_Bil"]

    # 特徴量2:AST/ALT比（De Ritis比 # 6/5
    df["AST_div_ALT_ex2"] = df["AST_GOT"] / df["ALT_GPT"]

    # 特徴量3:.フェリチン/AST比 フェリチンの代わりにタンパク質 7/6
    df["TP_div_AST_ex2"] = df["TP"] / df["AST_GOT"]

    # 特徴量4:.グロブリン  1/(8*9)
    df["Globulin_ex2"] = np.reciprocal(df["Alb"] / df["AG_ratio"])

    print("処理後:", df.shape)
    print("特徴量を生成しました(完了)")
    print("=" * 40)
    return df


# %%
# 特徴量作成 exp03
# =================================================
def data_pre02(df):
    # 特徴量1:ビリルビン/酵素比　総ビリルビン / ALT または 総ビリルビン / AST 2/5 2/6
    # df["TB_div_ALT_ex3"] = df["T_Bil"] / df["ALT_GPT"]
    # df["TB_div_AST_ex3"] = df["T_Bil"] / df["AST_GOT"]

    # 特徴量2: 総ビリルビン/ALP比  2/4
    df["TB_div_ALP_ex3"] = df["T_Bil"] / df["ALP"]

    # 特徴量3:アルブミン/ALT比 8/5
    df["Alb_div_ALT_ex3"] = df["Alb"] / df["ALT_GPT"]

    # 特徴量4:総タンパク/ALT比 7/5
    # df["TP_div_ALT_ex3"] = df["TP"] / df["ALT_GPT"]

    # 特徴量5:ALP/AST比またはALP/ALT比 4/6 4/5
    # df["ALP_div_AST_ex3"] = df["ALP"] / df["AST_GOT"]
    # df["ALP_div_ALT_ex3"] = df["ALP"] / df["ALT_GPT"]

    # 特徴量6:総ビリルビン / アルブミン 2/8
    # df["TB_div_Alb_ex3"] = df["T_Bil"] / df["Alb"]

    print("処理後:", df.shape)
    print("特徴量を生成しました(完了)")
    print("=" * 40)
    return df


# %%
# 学習関数の定義
# =================================================
def train_lgb(
    input_x,
    input_y,
    input_id,
    params,
    list_nfold=[0, 1, 2, 3, 4],
    n_splits=5,
):
    metrics = []
    imp = pd.DataFrame()
    train_oof = np.zeros(len(input_x))

    # cross-validation
    cv = list(
        StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123).split(
            input_x, input_y
        )
    )

    # 1.学習データと検証データに分離
    for nfold in list_nfold:
        print("-" * 20, nfold, "-" * 20)
        print(dt_now().strftime("%Y年%m月%d日 %H:%M:%S"))

        idx_tr, idx_va = cv[nfold][0], cv[nfold][1]

        x_tr, y_tr = (
            input_x.loc[idx_tr, :],
            input_y[idx_tr],
        )
        x_va, y_va = (
            input_x.loc[idx_va, :],
            input_y[idx_va],
        )

        print(x_tr.shape, x_va.shape)

        # モデルの保存先名
        fname_lgb = os.path.join(EXP_MODEL, f"model_lgb_fold{nfold}.pickle")

        if not os.path.isfile(
            os.path.join(EXP_MODEL, f"model_lgb_fold{nfold}.pickle")
        ):  # if trained model, no training
            # train
            print("-------training start-------")
            model = lgb.LGBMClassifier(**params)
            model.fit(
                x_tr,
                y_tr,
                eval_set=[(x_tr, y_tr), (x_va, y_va)],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=100, verbose=True),
                    lgb.log_evaluation(100),
                ],
            )

            # モデルの保存
            with open(fname_lgb, "wb") as f:
                pickle.dump(model, f, protocol=4)

        else:
            print("すでに学習済みのためモデルを読み込みます")
            with open(fname_lgb, "rb") as f:
                model = pickle.load(f)

        # evaluate
        y_tr_pred = model.predict_proba(x_tr)[:, 1]
        y_va_pred = model.predict_proba(x_va)[:, 1]
        metric_tr = roc_auc_score(y_tr, y_tr_pred)
        metric_va = roc_auc_score(y_va, y_va_pred)
        metrics.append([nfold, metric_tr, metric_va])
        print(f"[auc] tr:{metric_tr:.4f}, va:{metric_va:.4f}")

        # oof
        train_oof[idx_va] = y_va_pred

        # imp
        _imp = pd.DataFrame(
            {"col": input_x.columns, "imp": model.feature_importances_, "nfold": nfold}
        )
        imp = pd.concat([imp, _imp])

    print("-" * 20, "result", "-" * 20)

    # metric
    metrics = np.array(metrics)
    print(metrics)
    print(f"[cv] tr:{metrics[:,1].mean():.4f}+-{metrics[:,1].std():.4f}, \
        va:{metrics[:,2].mean():.4f}+-{metrics[:,1].std():.4f}")

    oof = f"[oof]{roc_auc_score(input_y, train_oof):.4f}"

    print(oof)

    # oof
    train_oof = pd.concat(
        [
            input_id,
            pd.DataFrame({"pred": train_oof}),
        ],
        axis=1,
    )

    # importance
    imp = imp.groupby("col")["imp"].agg(["mean", "std"]).reset_index(drop=False)
    imp.columns = ["col", "imp", "imp_std"]

    # stdout と stderr を一時的にリダイレクト
    stdout_logger = logging.getLogger("STDOUT")
    stderr_logger = logging.getLogger("STDERR")

    sys_stdout_backup = sys.stdout
    sys_stderr_backup = sys.stderr

    sys.stdout = StreamToLogger(stdout_logger, logging.INFO)
    sys.stderr = StreamToLogger(stderr_logger, logging.ERROR)
    print("-" * 20, "result", "-" * 20)
    print(dt_now().strftime("%Y年%m月%d日 %H:%M:%S"))
    print(name)
    print(input_x.shape)
    print(metrics)
    print(f"[cv] tr:{metrics[:,1].mean():.4f}+-{metrics[:,1].std():.4f}, \
        va:{metrics[:,2].mean():.4f}+-{metrics[:,1].std():.4f}")

    print(oof)

    print("-" * 20, "importance", "-" * 20)
    print(imp.sort_values("imp", ascending=False)[:10])

    # リダイレクトを解除
    sys.stdout = sys_stdout_backup
    sys.stderr = sys_stderr_backup

    return train_oof, imp, metrics


# %%
# 推論関数の定義 =================================================
def predict_lgb(
    input_x,
    input_id,
    list_nfold=[0, 1, 2, 3, 4],
):
    pred = np.zeros((len(input_x), len(list_nfold)))

    for nfold in list_nfold:
        print("-" * 20, nfold, "-" * 20)

        fname_lgb = os.path.join(EXP_MODEL, f"model_lgb_fold{nfold}.pickle")
        with open(fname_lgb, "rb") as f:
            model = pickle.load(f)

        # 推論
        pred[:, nfold] = model.predict_proba(input_x)[:, 1]

    # 平均値算出
    pred = pd.concat(
        [
            input_id,
            pd.DataFrame(pred.mean(axis=1)),
        ],
        axis=1,
    )
    print("Done.")

    return pred


# %% setup
# set up
# =================================================
# utils
warnings.filterwarnings("ignore")
sns.set(font="IPAexGothic")
##!%matplotlib inline
pd.options.display.float_format = "{:10.4f}".format  # 表示桁数の設定

# フォルダの作成
make_dirs()
# ファイルの確認
file_list = file_list(INPUT_PATH)

# utils
# ログファイルの設定
logging.basicConfig(
    filename=f"{OUTPUT_EXP}/log_{name}.txt", level=logging.INFO, format="%(message)s"
)
# ロガーの作成
logger = logging.getLogger()


# 出力表示数増やす
# pd.set_option('display.max_rows',None)
# pd.set_option('display.max_columns',None)


# %% ファイルの読み込み
# Load Data
# =================================================
# train
df_train = load_data(2)
# %%
display(df_train.shape)
df_train.info()


# %%
df_train.describe().T
# %%
df_train.describe(include="O").T

# %%
# これまでの処理
# ==========================================================
df_train = data_pre01(df_train)
df_train = data_pre02(df_train)
# %% [markdown]
## Main 分析start!
# =========================================================
# %%
df_train.head()


# %%
# sns.histplot(data=df_train, x="Age")

sns.histplot(data=df_train, x="Age", hue="disease", bins=30)

# %%☆
# 0:年齢を離散化
# =========================================================

df_train["Cate_Age"] = pd.cut(df_train["Age"].values, 8)
display(pd.crosstab(df_train["Cate_Age"], df_train["disease"]))

display(pd.crosstab(df_train["Cate_Age"], df_train["disease"], normalize="index"))


# %%
# スライス、年と性別
pd.pivot_table(
    data=df_train, index="Cate_Age", columns="Gender", values="disease", aggfunc="mean"
)
# %%
sns.countplot(data=df_train, x="Gender", hue="disease")
display(pd.crosstab(df_train["Gender"], df_train["disease"], normalize="index"))
# %%
# 2:T_Bil
# =========================================================
sns.histplot(data=df_train, x="T_Bil", hue="disease")

# %%
sns.violinplot(data=df_train, x="T_Bil", hue="disease")
# %%
sns.violinplot(data=df_train, x="T_Bil")

# %%
df_train['log_T_Bil'] = np.log(df_train['T_Bil'])
sns.histplot(data=df_train, x="log_T_Bil", hue="disease")


#%%
df_train["T_Bil_0.7over"] = [i if i < 0.7 else 0.7 for i in df_train["log_T_Bil"]]

sns.histplot(data=df_train, x="T_Bil_0.7over", hue="disease")

# %%☆
df_train["Cate_T_Bil_0.7over"] = pd.cut(df_train["T_Bil_0.7over"], 20)
display(pd.crosstab(df_train["Cate_T_Bil_0.7over"], df_train["disease"]))

display(
    pd.crosstab(df_train["Cate_T_Bil_0.7over"], df_train["disease"], normalize="index")
)

# %%
# df_train["Cate_T_Bil_2.5over"] =pd.cut(df_train["T_Bil_2.5over"],bins=[0,1,1.75,2.5])
# display(pd.crosstab(df_train["Cate_T_Bil_2.5over"],df_train["disease"]))

# display(pd.crosstab(df_train["Cate_T_Bil_2.5over"],df_train["disease"],normalize="index"))
# %%
# 3:D_Bil
# =========================================================

sns.histplot(data=df_train, x="D_Bil", hue="disease")

#%%
df_train['log_D_Bil'] = np.log(df_train['D_Bil'])
sns.histplot(data=df_train, x="log_D_Bil", hue="disease")

# %%
df_train["log_D_Bil_0over"] = [i if i < 0 else 0 for i in df_train["log_D_Bil"]]


sns.histplot(data=df_train, x="log_D_Bil_0over", hue="disease")

# %%☆
df_train["Cate_log_D_Bil_0over"] = pd.cut(df_train["log_D_Bil_0over"], 20)
display(pd.crosstab(df_train["Cate_log_D_Bil_0over"], df_train["disease"]))

display(
    pd.crosstab(df_train["Cate_log_D_Bil_0over"], df_train["disease"], normalize="index")
)
# %%
# 4:ALP
# =========================================================
sns.histplot(data=df_train, x="ALP", hue="disease")
# %%
# df_train["CATE_ALP"] = pd.cut(df_train["ALP"], 4)
# df_train.groupby("CATE_ALP")["disease"].mean()
# %%
sns.violinplot(data=df_train["ALP"])

#%%
df_train['log_ALP'] = np.log(df_train['ALP'])
sns.histplot(data=df_train, x="log_ALP", hue="disease")

# %%
df_train["log_ALP_6.6over"] = [i if i < 6.6 else 6.6 for i in df_train["log_ALP"]]


sns.histplot(data=df_train, x="log_ALP_6.6over", hue="disease")
# %%☆
df_train["Cate_log_ALP_6.6over"] = pd.cut(df_train["log_ALP_6.6over"], 40)
display(pd.crosstab(df_train["Cate_log_ALP_6.6over"], df_train["disease"]))

display(
    pd.crosstab(df_train["Cate_log_ALP_6.6over"], df_train["disease"], normalize="index")
)
# %%☆
# 5:ALT_GPT
# =========================================================
sns.histplot(data=df_train, x="ALT_GPT", hue="disease")

#%%
df_train['log_ALT_GPT'] = np.log(df_train['ALT_GPT'])
sns.histplot(data=df_train, x="log_ALT_GPT", hue="disease")

# %%
df_train["log_ALT_GPT_5over"] = [i if i < 5 else 5 for i in df_train["log_ALT_GPT"]]


sns.histplot(data=df_train, x="log_ALT_GPT_5over", hue="disease")



# %%

df_train["Cate_log_ALT_GPT_5over"] = pd.cut(df_train["log_ALT_GPT_5over"], 20)
display(pd.crosstab(df_train["Cate_log_ALT_GPT_5over"], df_train["disease"]))

display(pd.crosstab(df_train["Cate_log_ALT_GPT_5over"], df_train["disease"],normalize="index"))


# %%☆
# 6:AST_GOT
# =========================================================
sns.histplot(data=df_train, x="AST_GOT", hue="disease")

#%%
df_train['log_AST_GOT'] = np.log(df_train['AST_GOT'])
sns.histplot(data=df_train, x="log_AST_GOT", hue="disease")

# %%
df_train["log_AST_GOT_5.5over"] = [i if i < 5.5 else 5.5 for i in df_train["log_AST_GOT"]]


sns.histplot(data=df_train, x="log_AST_GOT_5.5over", hue="disease")



# %%

df_train["Cate_log_AST_GOT_5.5over"] = pd.cut(df_train["log_AST_GOT_5.5over"], 30)
display(pd.crosstab(df_train["Cate_log_AST_GOT_5.5over"], df_train["disease"]))

display(pd.crosstab(df_train["Cate_log_AST_GOT_5.5over"], df_train["disease"],normalize="index"))

# %%


# %%
df_train.info()
# %%
