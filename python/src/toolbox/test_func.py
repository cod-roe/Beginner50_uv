

import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as mae

def data_pre(df):
  nonnull_list = []
  for col in df.columns:
    nonnull = df[col].count()
    if nonnull == 0:
      nonnull_list.append(col)
  nonnull_list


  df = df.drop(nonnull_list,axis=1)

  df = df.drop('市区町村名', axis=1)

  df = df.drop('種類',axis=1)

  dis = {
      '30分?60分':45,
      '1H30?2H':105,
      '1H?1H30':75,
      '2H?':120
  }

  df['最寄駅：距離（分）'] = df['最寄駅：距離（分）'].replace(dis).astype(float)

  df['面積（㎡）'] = df['面積（㎡）'].replace('2000㎡以上',2000).astype(float)

  y_list = {}
  for i in df['建築年'].value_counts().keys():
    if '平成' in i:
      num = float(i.split('平成')[1].split('年')[0])
      year = 36- num
    if '令和' in i:
      num = float(i.split('令和')[1].split('年')[0])
      year = 6- num
    if '昭和' in i:
      num = float(i.split('昭和')[1].split('年')[0])
      year = 99- num
    y_list[i] = year
  y_list['戦前'] = 79
  df['建築年'] = df['建築年'].replace(y_list)

  
  year = {
      '年第１四半期':'.25',
      '年第２四半期':'.50',
      '年第３四半期':'.75',
      '年第４四半期':'.99',
  }
  year_list = {}
  for i in df['取引時点'].value_counts().keys():
    for k,j in year.items():
      if k in i:
        year_rep = i.replace(k,j)
      year_list[i] = year_rep
  df['取引時点'] = df['取引時点'].replace(year_list).astype(float) 

  for col in ['都道府県名','地区名','最寄駅：名称','間取り','建物の構造','今後の利用目的','用途','都市計画','改装','取引の事情等']:
    df[col] = df[col].astype('category')

  return df



def model_lgb(df):
  df_train,df_val = train_test_split(df,test_size=0.2)

  col = '取引価格（総額）_log'
  train_y = df_train[col]
  train_x = df_train.drop(col,axis=1)

  col = '取引価格（総額）_log'
  val_y = df_val[col]
  val_x = df_val.drop(col,axis=1)

  trains = lgb.Dataset(train_x, train_y)
  valids = lgb.Dataset(val_x, val_y)

  params = {
      'objective':'regression',
      'metrics':'mae'
  }

  model = lgb.train(params,trains,valid_sets=valids,num_boost_round=1000,callbacks=[lgb.early_stopping(stopping_rounds=100)])

  return model

