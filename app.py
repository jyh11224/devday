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
    unique_codes = df_combined['발전소코드'].unique()[:num_plots]
    num_rows = (num_plots - 1) // 3 + 1
    num_cols = min(num_plots, 3)

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 12), sharex=True, sharey=True)

    for i, code in enumerate(unique_codes):
        row = i // num_cols
        col = i % num_cols

        df_subset = df_combined[df_combined['발전소코드'] == code]

        axs[row, col].plot(df_subset.index, df_subset['실측'])
        axs[row, col].set_title(f'발전소 Code: {code}')

    plt.xlabel('datetime')
    plt.tight_layout()
    st.pyplot(fig)

def main():
    # Read and preprocess data
    df_combined = read_data()

    # Streamlit App
    st.title('Power Plant Time Series Visualization')
    
    # Show data table if needed
    # st.dataframe(df_combined)

    # Plot power plants
    num_plots = st.slider('Number of Power Plants to Plot', min_value=1, max_value=9, value=3, step=1)
    plot_power_plants(df_combined, num_plots)

if __name__ == '__main__':
    main()
