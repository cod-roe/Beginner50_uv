# %% [markdown]
## スタッキング！
# =================================================
# svm、tf,lgbm

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
import random
from IPython.display import display
import warnings
# import zipfile

import numpy as np
import pandas as pd


# 可視化
import matplotlib.pyplot as plt
import seaborn as sns
# import japanize_matplotlib


# lightGBM
import lightgbm as lgb

# sckit-learn
# from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder

from sklearn.model_selection import StratifiedKFold  # ,train_test_split, KFold
from sklearn.metrics import roc_auc_score  # accuracy_score,confusion_matrix

import tensorflow as tf
import tensorflow.python.keras.backend as K
from tensorflow.keras.models import Sequential, Model  # type:ignore
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization  # type:ignore
from tensorflow.keras.layers import Embedding, Flatten, Concatenate  # type:ignore
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    LearningRateScheduler,
)  # type:ignore
from tensorflow.keras.optimizers import Adam, SGD  # type:ignore
from tensorflow.keras.metrics import AUC


from sklearn.svm import SVC
from sklearn.linear_model import Lasso

# %%
# Config
# =================================================

######################
# serial #
######################
serial_number = 15  # スプレッドシートAの番号

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
# lgbmチューニング
params = {
    "num_leaves": 116,
    "min_child_samples": 20,
    "min_sum_hessian_in_leaf": 0.000746344763289229,
    "feature_fraction": 0.722388190251776,
    "bagging_fraction": 0.8786978558889691,
    "lambda_l1": 0.18669365423501472,
    "lambda_l2": 0.18142925705123572,
    "boosting_type": "gbdt",
    "objective": "binary",
    "metric": "auc",
    "verbosity": -1,
    "learning_rate": 0.05,
    "n_estimators": 100000,
    "bagging_freq": 1,
    "random_state": 123,
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
# 特徴量生成 exp002
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
# lgbm
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
# tfモデル作成
def create_model():
    input_num = Input(shape=(16,))
    x_num = Dense(128, activation="relu")(input_num)
    x_num = BatchNormalization()(x_num)
    x_num = Dropout(0.3)(x_num)
    x_num = Dense(128, activation="relu")(x_num)
    x_num = BatchNormalization()(x_num)
    x_num = Dropout(0.2)(x_num)
    x_num = Dense(128, activation="relu")(x_num)
    x_num = BatchNormalization()(x_num)
    x_num = Dropout(0.1)(x_num)
    out = Dense(1, activation="sigmoid")(x_num)

    model = Model(
        inputs=input_num,
        outputs=out,
    )

    model.compile(
        optimizer="Adam",
        loss="binary_crossentropy",
        metrics=[AUC(name="roc_auc")],
    )

    return model


# tf用学習関数
def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    session_conf = tf.compat.v1.ConfigProto(
        intra_op_parallelism_threads=1, inter_op_parallelism_threads=1
    )
    sess = tf.compat.v1.Session(
        graph=tf.compat.v1.get_default_graph(), config=session_conf
    )
    # tf.compat.v1.keras.backend.set_session(sess)
    K.set_session(sess)


def train_tf(
    input_x,
    input_y,
    input_id,
    list_nfold=[0, 1, 2, 3, 4],
    batch_size=8,
    epochs=10000,
):
    # 推論値を格納する変数の作成
    train_oof = np.zeros(len(input_x))
    # 評価値を入れる変数の作成
    metrics = []

    # validation
    cv = list(
        StratifiedKFold(n_splits=5, shuffle=True, random_state=123).split(
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

        # 保存するモデルのファイル名
        fname_tf = os.path.join(EXP_MODEL, f"model_tf_fold{nfold}.weights.h5")

        if not os.path.isfile(fname_tf):
            # if mode_train == 'train':
            print("trainning start!")
            seed_everything(seed=123)
            model = create_model()
            model.fit(
                x=x_tr,
                y=y_tr,
                validation_data=(x_va, y_va),
                batch_size=batch_size,
                epochs=epochs,
                callbacks=[
                    ModelCheckpoint(
                        filepath=fname_tf,
                        monitor="val_loss",
                        mode="min",
                        verbose=1,
                        save_best_only=True,
                        save_weights_only=True,
                    ),
                    EarlyStopping(
                        monitor="val_loss",
                        mode="min",
                        min_delta=0,
                        patience=10,
                        verbose=1,
                        restore_best_weights=True,
                    ),
                    ReduceLROnPlateau(
                        monitor="val_loss",
                        mode="min",
                        factor=0.1,
                        patience=5,
                        verbose=1,
                    ),
                ],
                verbose=1,
            )

        else:
            print("model load.")
            model = create_model()
            model.load_weights(fname_tf)

        # validの推論値取得
        y_tr_pred = model.predict(x_tr)
        y_va_pred = model.predict(x_va)

        metric_tr = roc_auc_score(y_tr, y_tr_pred)
        metric_va = roc_auc_score(y_va, y_va_pred)
        metrics.append([nfold, metric_tr, metric_va])
        print(f"[auc] tr:{metric_tr:.4f}, va:{metric_va:.4f}")

        # oof
        train_oof[idx_va] = y_va_pred.ravel()

    print("-" * 10, "result", "-" * 10)
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

    # リダイレクトを解除
    sys.stdout = sys_stdout_backup
    sys.stderr = sys_stderr_backup

    return train_oof, metrics


# %%
# svm用学習関数
def train_svm(
    input_x,
    input_y,
    input_id,
    list_nfold=[0, 1, 2, 3, 4],
    n_splits=5,
):
    metrics = []
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
        fname_svm = os.path.join(EXP_MODEL, f"model_svm_fold{nfold}.pickle")

        if not os.path.isfile(fname_svm):  # if trained model, no training
            # train
            print("-------training start-------")
            model_svm = SVC(C=1.0, random_state=123, probability=True)
            model_svm.fit(x_tr, y_tr)

            # モデルの保存
            with open(fname_svm, "wb") as f:
                pickle.dump(model_svm, f, protocol=4)

        else:
            print("すでに学習済みのためモデルを読み込みます")
            with open(fname_svm, "rb") as f:
                model_svm = pickle.load(f)

        # evaluate
        y_tr_pred = model_svm.predict_proba(x_tr)[:, 1]
        y_va_pred = model_svm.predict_proba(x_va)[:, 1]
        metric_tr = roc_auc_score(y_tr, y_tr_pred)
        metric_va = roc_auc_score(y_va, y_va_pred)
        metrics.append([nfold, metric_tr, metric_va])
        print(f"[auc] tr:{metric_tr:.4f}, va:{metric_va:.4f}")

        # oof
        train_oof[idx_va] = y_va_pred

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

    # リダイレクトを解除
    sys.stdout = sys_stdout_backup
    sys.stderr = sys_stderr_backup

    return train_oof, metrics


# %% setup
# set up
# =================================================
# utils
warnings.filterwarnings("ignore")
sns.set(font="IPAexGothic")
###!%matplotlib inline
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
df_train.info()


# %%
# データセット作成
# =================================================
set_file = df_train
x_train = set_file.drop([target_columns], axis=1)
y_train = set_file[target_columns]
id_train = pd.DataFrame(set_file.index)

print(x_train.shape, y_train.shape, id_train.shape)

# %%
# x_train = data_pre00(x_train)
train_oof_lgbm, imp_lgbm, metrics_lgbm = train_lgb(
    x_train,
    y_train,
    id_train,
    params,
    list_nfold=[0, 1, 2, 3, 4],
    n_splits=5,
)

# %%
# 正規化
dict_num = {}
for col in x_train.columns:
    value_min = x_train[col].min()
    value_max = x_train[col].max()
    x_train[col] = (x_train[col] - value_min) / (value_max - value_min)

    # testデータにも適用できるように保存
    dict_num[col] = {}
    dict_num[col]["min"] = value_min
    dict_num[col]["max"] = value_max

x_train.describe()

# %%


# %%
# tf学習
model = create_model()
model.summary()
train_oof_tf, metrics_tf = train_tf(
    x_train,
    y_train,
    id_train,
    list_nfold=[0, 1, 2, 3, 4],
    batch_size=8,
    epochs=10000,
)

# %%
# svm学習の実行
train_oof_svm, metrics_svm = train_svm(
    x_train,
    y_train,
    id_train,
    list_nfold=[0, 1, 2, 3, 4],
    n_splits=5,
)
# %%
df_train_las = pd.DataFrame(
    {
        "pred1": train_oof_lgbm["pred"],
        "pred2": train_oof_tf["pred"],
        "pred3": train_oof_svm["pred"],
        "true": y_train,
    }
)
# データセット
x_train_las = df_train_las[["pred1", "pred2", "pred3"]]

# %%
oof_las = np.zeros(len(x_train_las))
models = []

cv = list(
    StratifiedKFold(n_splits=5, shuffle=True, random_state=123).split(
        x_train_las, y_train
    )
)

# 1.学習データと検証データに分離
for nfold in np.arange(5):
    print("-" * 20, nfold, "-" * 20)
    print(dt_now().strftime("%Y年%m月%d日 %H:%M:%S"))

    idx_tr, idx_va = cv[nfold][0], cv[nfold][1]

    x_tr, y_tr = (
        x_train_las.loc[idx_tr, :],
        y_train[idx_tr],
    )
    x_va, y_va = (
        x_train_las.loc[idx_va, :],
        y_train[idx_va],
    )

    print(x_tr.shape, x_va.shape)

    # # モデルの保存先名
    # fname_las = os.path.join(EXP_MODEL, f"model_las_fold{nfold}.pickle")

    # if not os.path.isfile(
    #     os.path.join(EXP_MODEL, f"model_las_fold{nfold}.pickle")
    # ):
    # if trained model, no training
    # train
    print("-------training start-------")
    model = Lasso(alpha=0.01)
    model.fit(x_tr, y_tr)
    models.append(model)

    #     # モデルの保存
    #     with open(fname_las, "wb") as f:
    #         pickle.dump(model, f, protocol=4)

    # else:
    #     print("すでに学習済みのためモデルを読み込みます")
    #     with open(fname_las, "rb") as f:
    #         model = pickle.load(f)

    # evaluate
    y_va_pred = model.predict(x_va)
    oof_las[idx_va] = y_va_pred

df_train_las["pred_ensemble3"] = oof_las
df_train_las["pred_ensemble3"] = df_train_las["pred_ensemble3"].clip(lower=0, upper=1)
df_train_las[["true", "pred1", "pred2", "pred3", "pred_ensemble3"]].head()


# %%
def evaluate_ensemble(input_df, col_pred):
    print(
        f'[auc] model1:{roc_auc_score(input_df["true"], input_df["pred1"]):.4f}, model2:{roc_auc_score(input_df["true"], input_df["pred2"]):.4f}, model3:{roc_auc_score(input_df["true"], input_df["pred3"]):.4f} -> ensemble:{roc_auc_score(input_df["true"], input_df[col_pred]):.4f}'
    )


evaluate_ensemble(df_train_las, col_pred="pred_ensemble3")
# %%
# テストファイルの読み込み
# =================================================
df_test = load_data(1)

# %%
df_test = data_pre01(df_test)
df_test = data_pre02(df_test)


# %%
# 推論データのデータセット作成
set_file = df_test
x_test = set_file
id_test = pd.DataFrame(set_file.index)

print(x_test.shape, id_test.shape)


# %%
# 推論関数の定義 lgbm
# =================================================
def predict_lgb(
    input_x,
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
    pred = pd.DataFrame(pred.mean(axis=1))
    print("Done.")

    return pred


test_pred_lgb = predict_lgb(
    x_test,
    list_nfold=[0, 1, 2, 3, 4],
)


# 欠損値補間・正規化

for col in x_test.columns:
    value_min = dict_num[col]["min"]
    value_max = dict_num[col]["max"]
    x_test[col] = (x_test[col] - value_min) / (value_max - value_min)

x_test.head()


# %%
# tf推論処理
def predict_tf(
    input_x,
    list_nfold=[0, 1, 2, 3, 4],
):
    pred = np.zeros((len(input_x), len(list_nfold)))

    for nfold in list_nfold:
        print("-" * 20, nfold, "-" * 20)

        fname_tf = os.path.join(EXP_MODEL, f"model_tf_fold{nfold}.weights.h5")
        model = create_model()
        model.load_weights(fname_tf)

        # 推論
        pred[:, nfold] = model.predict(input_x).ravel()

    # 平均値算出
    pred = pd.DataFrame(pred.mean(axis=1))
    print("Done.")

    return pred


test_pred_tf = predict_tf(
    x_test,
    list_nfold=[0, 1, 2, 3, 4],
)


# %%svm推論
def predict_svm(
    input_x,
    list_nfold=[0, 1, 2, 3, 4],
):
    pred = np.zeros((len(input_x), len(list_nfold)))

    for nfold in list_nfold:
        print("-" * 20, nfold, "-" * 20)

        fname_svm = os.path.join(EXP_MODEL, f"model_svm_fold{nfold}.pickle")
        with open(fname_svm, "rb") as f:
            model = pickle.load(f)

        # 推論
        pred[:, nfold] = model.predict_proba(input_x)[:, 1]

    # 平均値算出
    pred = pd.DataFrame(pred.mean(axis=1))
    print("Done.")

    return pred


test_pred_svm = predict_svm(
    x_test,
    list_nfold=[0, 1, 2, 3, 4],
)

# %%
# %%


# %%
df_test_las = pd.concat([test_pred_lgb, test_pred_tf, test_pred_svm], axis=1)
df_test_las.columns = ["pred1", "pred2", "pred3"]
# df_test_las = pd.DataFrame(
#     {
#         "pred1": test_pred_lgb["pred"],
#         "pred2": test_pred_tf["pred"],
#         "pred3": test_pred_svm["pred"],
#     }
# )
df_test_las.head()
# %%

df_test_las["pred_ensemble3"] = 0

for model in models:
    df_test_las["pred_ensemble3"] += model.predict(
        df_test_las[["pred1", "pred2", "pred3"]]
    ) / len(models)

df_test_las["pred_ensemble3"] = df_test_las["pred_ensemble3"].clip(lower=0, upper=1)
# %%
df_test_las["pred_ensemble3"].head()
# %%

test_pred = pd.concat(
    [
        id_test,
        df_test_las["pred_ensemble3"],
    ],
    axis=1,
)

print(test_pred.shape)
test_pred.head(15)
# 提出データの作成
# %%
if not skip_run:
    test_pred.to_csv(
        os.path.join(OUTPUT_EXP, f"submission_{name}.csv"), index=False, header=False
    )


# %%
