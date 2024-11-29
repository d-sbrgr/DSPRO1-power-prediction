from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from . import get_project_root
from .util import df_time_to_utc_plus_one


def read_formatted_source_data() -> pd.DataFrame:
    """Reads the input data from the input file."""
    file_path = get_project_root() / 'data' / 'formatted_source_data.parquet'
    return pd.read_parquet(file_path, engine='pyarrow')


def read_cleaned_data_retain_nan() -> pd.DataFrame:
    """Reads the input data from the input file."""
    file_path = get_project_root() / 'data' / 'cleaned_strategy_1.parquet'
    return pd.read_parquet(file_path, engine='pyarrow')


def read_cleaned_data_remove_nan() -> pd.DataFrame:
    """Reads the input data from the input file."""
    file_path = get_project_root() / 'data' / 'cleaned_strategy_2.parquet'
    return pd.read_parquet(file_path, engine='pyarrow')


def read_cleaned_data_interpolate_nan() -> pd.DataFrame:
    """Reads the input data from the input file."""
    file_path = get_project_root() / 'data' / 'cleaned_strategy_3.parquet'
    return pd.read_parquet(file_path, engine='pyarrow')


def read_time_decomposition_remainder_data() -> pd.DataFrame:
    """Reads the input data from the input file."""
    file_path = get_project_root() / 'data' / 'time_decomposition_remainder_data.parquet'
    return pd.read_parquet(file_path, engine='pyarrow')


def convert_date(df) -> pd.DataFrame:
    """Converts the date column to datetime format and adds the features Year, Month, Day, Hour and Weekday."""
    df['Date'] = pd.to_datetime(df['Date'])
    df = df_time_to_utc_plus_one(df)
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['Hour'] = df['Date'].dt.hour
    df['Weekday'] = df['Date'].dt.weekday + 1
    return df

def combine_power_consumption(df) -> None:
    """Combines the power consumption columns."""
    df['NE_tot'] = df['Value_NE5'] + df['Value_NE7']

def clean_time_sync(df) -> pd.DataFrame:
    """Removes the last rows with NaN values."""
    idx_drop = df[(df["NE_tot"].isna()) & (df["Date"] >= pd.to_datetime("2024-10-02"))].index
    df_time = df.drop(idx_drop)
    return df_time

def clean_humidity(df) -> None:
    """Cleans the humidity column."""
    df.loc[df['Hr [%Hr]'] > 100, 'Hr [%Hr]'] = 100

def add_corona_feature(df) -> None:
    """Adds the corona feature."""
    corona_period_1 = (df['Date'] >= '2020-03-18') & (df['Date'] <= '2020-06-06')
    corona_period_2 = (df['Date'] >= '2021-01-18') & (df['Date'] <= '2021-03-04')
    df['Corona'] = (corona_period_1 | corona_period_2).astype(int)

def plot_data(df) -> None:
    """Plots the data on the plot."""
    fig, axes = plt.subplots(3, 4, figsize=(15, 10))
    cols = ['Value_NE5', 'Value_NE7', 'Hr [%Hr]', 'RainDur [min]', 'StrGlo [W/m2]', 'T [°C]', 'WD [°]', 'WVs [m/s]',
            'WVv [m/s]', 'p [hPa]', 'NE_tot']
    for i, col in enumerate(cols):
        if col in df.columns:
            row_plot = i // 4
            col_plot = i % 4
            df[col].plot(ax=axes[row_plot, col_plot], title=col)

    plt.tight_layout()
    plt.show()

def clean_typ(df) -> pd.DataFrame:
    """Set the features Vacation, Holiday, Year, Month, Day, Hour, Weekday and Corona as categorical typ."""
    df['Vacation'] = df['Vacation'].astype('category')
    df['Holiday'] = df['Holiday'].astype('category')
    df['Year'] = df['Year'].astype('category')
    df['Month'] = df['Month'].astype('category')
    df['Day'] = df['Day'].astype('category')
    df['Hour'] = df['Hour'].astype('category')
    df['Weekday'] = df['Weekday'].astype('category')
    df['Corona'] = df['Corona'].astype('category')
    return df

def clean_retain_nan(df) -> pd.DataFrame:
    """NaN values are retained, and the implausible data are overwritten with NaN values."""
    df_strategy_1 = df.copy()
    df_strategy_1.loc[df_strategy_1['Value_NE5'] < 60000, 'Value_NE5'] = None
    df_strategy_1.loc[df_strategy_1['Value_NE7'] < 100000, 'Value_NE7'] = None
    combine_power_consumption(df_strategy_1)
    return df_strategy_1

def clean_remove_nan(df) -> pd.DataFrame:
    """Removes NaN values as well as the implausible values."""
    df_strategy_2 = df.copy()
    df_strategy_2 = df_strategy_2.dropna()
    df_strategy_2 = df_strategy_2.drop(df_strategy_2[df_strategy_2['Value_NE5'] < 60000].index)
    df_strategy_2 = df_strategy_2.drop(df_strategy_2[df_strategy_2['Value_NE7'] < 100000].index)
    combine_power_consumption(df_strategy_2)
    df_strategy_2.reset_index(drop=True, inplace=True)
    return df_strategy_2

def clean_interpolate_nan(df) -> pd.DataFrame:
    """NaN values as well as the implausible values are replaced by the mean. The mean is calculated from the values
    where the Month, Day, and Hour are identical."""
    df_strategy_3 = df.copy()
    columns = ['Value_NE5','Value_NE7', 'Hr [%Hr]', 'T [°C]', 'WD [°]', 'WVs [m/s]', 'WVv [m/s]', 'p [hPa]',
               'RainDur [min]', 'StrGlo [W/m2]']

    for column in columns:
        nan_rows = df_strategy_3[df_strategy_3[[column]].isna().any(axis=1)]
        mean_values = (df_strategy_3.groupby(['Month', 'Day', 'Hour'], observed=False)[column].mean().to_dict())
        for index, row in nan_rows.iterrows():
            month = row['Month']
            day = row['Day']
            hour = row['Hour']
            value = mean_values.get((month, day, hour))
            df_strategy_3.at[index, column] = value

    mask = df_strategy_3['Value_NE5'] < 60000
    mean_values = df_strategy_3.groupby(['Month', 'Day', 'Hour'], observed=False)['Value_NE5'].mean().to_dict()
    for index, row in df_strategy_3[mask].iterrows():
        month = row['Month']
        day = row['Day']
        hour = row['Hour']
        value = mean_values.get((month, day, hour))
        if value is not None:
            df_strategy_3.at[index, 'Value_NE5'] = value

    mask = df_strategy_3['Value_NE7'] < 100000
    mean_values = df_strategy_3.groupby(['Month', 'Day', 'Hour'], observed=False)['Value_NE7'].mean().to_dict()
    for index, row in df_strategy_3[mask].iterrows():
        month = row['Month']
        day = row['Day']
        hour = row['Hour']
        value = mean_values.get((month, day, hour))
        if value is not None:
            df_strategy_3.at[index, 'Value_NE7'] = value
    combine_power_consumption(df_strategy_3)
    return df_strategy_3

def get_data_retain_nan() -> pd.DataFrame:
    """Returns the DataFrame with the cleaning strategy 1."""
    df = read_formatted_source_data()
    df = convert_date(df)
    combine_power_consumption(df)
    df_time = clean_time_sync(df)
    clean_humidity(df_time)
    add_corona_feature(df_time)
    df_type = clean_typ(df_time)
    return clean_retain_nan(df_type)

def get_data_remove_nan() -> pd.DataFrame:
    """Returns the DataFrame with the cleaning strategy 2."""
    df = read_formatted_source_data()
    df = convert_date(df)
    combine_power_consumption(df)
    df_time = clean_time_sync(df)
    clean_humidity(df_time)
    add_corona_feature(df_time)
    df_type = clean_typ(df_time)
    return clean_remove_nan(df_type)

def get_data_interpolate_nan() -> pd.DataFrame:
    """Returns the DataFrame with the cleaning strategy 3."""
    df = read_formatted_source_data()
    df = convert_date(df)
    combine_power_consumption(df)
    df_time = clean_time_sync(df)
    clean_humidity(df_time)
    add_corona_feature(df_time)
    df_type = clean_typ(df_time)
    return clean_interpolate_nan(df_type)
