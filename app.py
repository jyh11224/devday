# sudo apt-get install fonts-nanum*
import matplotlib 
# matplotlib.font_manager._rebuild()
# sudo rm -rf ~/.cache/

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Function to read and preprocess data
def read_data():
    df = pd.read_csv('https://raw.githubusercontent.com/jyh11224/devday/main/%EC%A0%9C11%ED%9A%8C%20%EC%82%B0%EC%97%85%ED%86%B5%EC%83%81%EC%9E%90%EC%9B%90%EB%B6%80%20%EA%B3%B5%EA%B3%B5%EB%8D%B0%EC%9D%B4%ED%84%B0%ED%99%9C%EC%9A%A9%20BI%EA%B3%B5%EB%AA%A8%EC%A0%84_%EB%B9%85%EB%8D%B0%EC%9D%B4%ED%84%B0%20%EB%B6%84%EC%84%9D%20%EA%B3%BC%EC%A0%9C%205_%EB%8D%B0%EC%9D%B4%ED%84%B0.csv')
    df = df.sort_values(by=['일자', '발전소코드'])
    df = df.replace('-', 0)

    dtype_mapping = {
        '2:00': float,
        '3:00': float,
        '4:00': float,
        '20:00': float,
        '21:00': float,
        '22:00': float,
        '23:00': float
    }
    df = df.astype(dtype_mapping)

    df['date'] = pd.to_datetime(df['일자'])
    time_columns = df.columns[4:28]
    df2 = df.melt(id_vars=['일자', '구분1', '발전소코드', '발전소용량(KW)'], value_vars=time_columns, var_name='hour', value_name='value')
    df2['datetime'] = pd.to_datetime(df2['일자'].astype(str) + ' ' + df2['hour'], format='%Y-%m-%d %H:%M')

    df_combined = df2.pivot(index=['datetime', '발전소코드', '발전소용량(KW)'], columns='구분1', values='value').reset_index()
    df_combined['gap'] = abs((df_combined['실측'] - df_combined['예측'])) / df_combined['발전소용량(KW)'] * 100
    df_combined['actual_average'] = df_combined['실측']
    df_combined['predict_average'] = df_combined['예측']

    df_combined.set_index('datetime', inplace=True)

    return df_combined

# Function to plot time series for each power plant
def plot_power_plants(df_combined, num_plots):
    unique_codes = df_combined['발전소코드'].unique()
    num_rows = 3
    num_cols = 3

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 12), sharex=True, sharey=True)

    for i, code in enumerate(unique_codes[:num_plots]):
        row = i // num_cols
        col = i % num_cols

        df_subset = df_combined[df_combined['발전소코드'] == code]

        axs[row, col].plot(df_subset.index, df_subset['실측'])
        axs[row, col].set_title(f'발전소 Code: {code}')
        axs[row, col].tick_params(labelrotation=45)
        
    axs[2][1].set_xlabel('datetime')
    axs[1][0].set_ylabel('발전량')
    plt.tight_layout()
    st.pyplot(fig)

# 특성 스케일링
def stdscaler():
    hours = df_pivot.index
    scaler = StandardScaler()
    
    scaled_error_data = scaler.fit_transform(df_pivot)
    
    
    df_scaled = pd.DataFrame(scaled_error_data, columns=df_pivot.columns)
    df_scaled.set_index(hours, inplace=True)
    df_T_scaled = df_scaled.T


def main():
    # Read and preprocess data
    df_combined = read_data()

    # Streamlit App
    st.title('Power Plant Time Series Visualization')
    
    # Show data table if needed
    st.write(df_combined)

    # Plot power plants
    numplots = 9
    plot_power_plants(df_combined, numplots)

if __name__ == '__main__':
    main()

############################ 수정필요 ####################################
df_combined['hour'] = df_combined['datetime'].dt.hour
df_pivot = df_combined.pivot(index='datetime', columns='발전소코드', values='gap')
df_pivot = df_pivot.reset_index()
df_pivot["hour"] = df_pivot['datetime'].astype("datetime64").dt.hour

df_pivot = df_pivot.drop(columns=['datetime']).groupby('hour').mean()
st.write(df_pivot)

# 발전소 코드 리스트
plant_list = df_combined['발전소코드'].tolist()
plant_list = list(set(plant_list))
plant_list.sort()

# 특성 스케일링
hours = df_pivot.index
scaler = StandardScaler()

scaled_error_data = scaler.fit_transform(df_pivot)


df_scaled = pd.DataFrame(scaled_error_data, columns=df_pivot.columns)
df_scaled.set_index(hours, inplace=True)
df_T_scaled = df_scaled.T

max_clusters = 20

inertia = []
silhouette_scores = []
for k in range(2, max_clusters+1):
    e_kmeans = KMeans(n_clusters=k)
    e_kmeans.fit(df_T_scaled)
    inertia.append(e_kmeans.inertia_) # ELBOW

    # Silhouette 계수 계산
    s_labels = e_kmeans.labels_
    silhouette_avg = silhouette_score(df_T_scaled, s_labels)
    silhouette_scores.append(silhouette_avg)

fig, ax1 = plt.subplots()

ax1.plot(range(2, max_clusters+1), inertia, 'bo-')
ax1.set_xlabel('Number of Clusters')
ax1.set_ylabel('Inertia')
ax1.set_title('Elbow Method')

# Silhouette 그래프 출력
ax2 = ax1.twinx()
ax2.plot(range(2, max_clusters+1), silhouette_scores, 'rs-')
ax2.set_ylabel('Silhouette Score', color='r')
ax2.tick_params('y', colors='r')

st.pyplot(fig)


# K-means 알고리즘을 사용하여 클러스터링 모델 학습
kmeans = KMeans(n_clusters=10, init='k-means++')  # 클러스터 개수는 상황에 맞게 설정합니다
kmeans.fit(df_T_scaled)

km = TimeSeriesKMeans(n_clusters=7, metric="euclidean")

labels = km.fit_predict(df_T_scaled)
# 학습된 클러스터링 모델로 데이터를 예측 (각 데이터 포인트가 속한 클러스터를 예측)

cluster_count = math.ceil(math.sqrt(len(set(labels))))
# A good rule of thumb is choosing k as the square root of the number of points in the training data set in kNN

st.write(labels)
score = silhouette_score(df_T_scaled, labels)
st.write(score)

plot_count = math.ceil(math.sqrt(len(set(labels))))


fig, axs = plt.subplots(plot_count,plot_count,figsize=(10,10))
fig.suptitle('Clusters')
row_i=0
column_j=0
# For each label there is,
# plots every series with that label
for label in set(labels):
    cluster = []
    for i in range(len(labels)):
            if(labels[i]==label):
                axs[row_i, column_j].plot([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23], df_scaled[plant_list[i]].values ,c="gray",alpha=0.4)
                cluster.append(df_scaled[plant_list[i]].values)
    if len(cluster) > 0:
        axs[row_i, column_j].plot(np.average(np.vstack(cluster),axis=0),c="red")
    axs[row_i, column_j].set_title("Cluster "+str(row_i*plot_count+column_j))
    axs[row_i, column_j].set_xlim(0, 23)
    column_j+=1
    if column_j%plot_count == 0:
        row_i+=1
        column_j=0

st.pyplot(fig)


cluster_c = [len(labels[labels==i]) for i in range(len(set(labels)))]
cluster_n = ["Cluster "+str(i) for i in range(len(set(labels)))]
plt.figure(figsize=(15,10))
plt.title("Cluster Distribution for KMeans")
plt.bar(cluster_n,cluster_c)
st.bar_chart(cluster_c)

fancy_names_for_labels = [f"Cluster {label}" for label in labels]
p = pd.DataFrame(zip(plant_list,fancy_names_for_labels),columns=["plant code","Cluster"]).sort_values(by="Cluster").set_index("plant code")
p = p.sort_index()
####################################################################

