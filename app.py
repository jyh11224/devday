# 용도별 사용할 lib들은 하기와 같습니다.

# Native libraries
import os
import math

# 데이터 처리
import pandas as pd
import numpy as np
from numpy import where
from numpy import unique
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# 데이터 시각화
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


# 모델링을 위한 패키지
from sklearn.model_selection import train_test_split, cross_validate, cross_val_predict, KFold
from sklearn import metrics
# !pip install tslearn
import tslearn
from sklearn.metrics import silhouette_score
from tslearn.clustering import TimeSeriesKMeans
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# 한글 폰트 설정 (에러가 날시에는 아래 3줄 주석처리)
# !sudo apt-get install -y fonts-nanum
# !sudo fc-cache -fv
# !rm ~/.cache/matplotlib -rf
# plt.rc('font', family='NanumGothic')


# warnings 무시
import os
import warnings
warnings.filterwarnings(action='ignore')

# streamlit
import streamlit as st

all_mySeries = []
actual_mySeries = []
predict_mySeries = []

df = pd.read_csv('https://raw.githubusercontent.com/jyh11224/devday/main/%EC%A0%9C11%ED%9A%8C%20%EC%82%B0%EC%97%85%ED%86%B5%EC%83%81%EC%9E%90%EC%9B%90%EB%B6%80%20%EA%B3%B5%EA%B3%B5%EB%8D%B0%EC%9D%B4%ED%84%B0%ED%99%9C%EC%9A%A9%20BI%EA%B3%B5%EB%AA%A8%EC%A0%84_%EB%B9%85%EB%8D%B0%EC%9D%B4%ED%84%B0%20%EB%B6%84%EC%84%9D%20%EA%B3%BC%EC%A0%9C%205_%EB%8D%B0%EC%9D%B4%ED%84%B0.csv')
df = df.sort_values(by=['일자', '발전소코드'])

## replace '-'
df = df.replace('-', 0)

# 열별로 데이터형 변환할 딕셔너리 생성
dtype_mapping = {
    '2:00': float,
    '3:00': float,
    '4:00': float,
    '20:00': float,
    '21:00': float,
    '22:00': float,
    '23:00': float
}

# 여러 개의 열을 한꺼번에 데이터형 변환
df = df.astype(dtype_mapping)

time_columns = df.columns[4:28]
time_columns

# 일자와 시간의 값을 합치고 datetime 컬럼 생성
df['date'] = pd.to_datetime(df['일자'])
time_columns = df.columns[4:28]
df2 = df.melt(id_vars=['일자', '구분1', '발전소코드', '발전소용량(KW)'], value_vars=time_columns, var_name='hour', value_name='value')
df2['datetime'] = pd.to_datetime(df2['일자'].astype(str) + ' ' + df2['hour'], format='%Y-%m-%d %H')

df_combined = df2.pivot(index=['datetime', '발전소코드', '발전소용량(KW)'], columns='구분1', values='value').reset_index()
df_combined['gap'] = abs((df_combined['실측'] - df_combined['예측'])) / df_combined['발전소용량(KW)'] * 100
df_combined['actual_average'] = df_combined['실측']
df_combined['predict_average'] = df_combined['예측']

df_combined.set_index('datetime')

unique_codes = df_combined['발전소코드'].unique()
num_plots = 9
num_rows = 3
num_cols = 3

fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 12), sharex=True, sharey=True)

for i, code in enumerate(unique_codes[:num_plots]):
    row = i // num_cols
    col = i % num_cols
    df_subset = df_combined[df_combined['발전소코드'] == code]
    axs[row, col].plot(df_subset['datetime'], df_subset['실측'])
    axs[row, col].set_title(f'발전소 Code: {code}')
    axs[row, col].tick_params(labelrotation=45)



axs[2][1].set_xlabel('datetime')
axs[1][0].set_ylabel('발전량')
plt.tight_layout()
st.pyplot(fig)
