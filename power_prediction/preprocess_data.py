from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def get_project_root() -> Path:
    """Returns the root directory of the project."""
    try:
        project_root = Path(__file__).parent.parent
    except NameError:
        project_root = Path().resolve()
    return project_root


def read_input() -> pd.DataFrame:
    """Reads the input data from the input file."""
    project_root = get_project_root()
    file_path = project_root / 'data' / 'formatted_source_data.csv'
    return pd.read_csv(file_path)


def convert_date(df) -> None:
    """Converts the date column to datetime format and adds the features Year, Month, Day, Hour and Weekday."""
    df['Date_format'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date_format'].dt.year
    df['Month'] = df['Date_format'].dt.month
    df['Day'] = df['Date_format'].dt.day
    df['Hour'] = df['Date_format'].dt.hour
    df['Weekday'] = df['Date_format'].dt.weekday + 1

def combine_power_consumption(df) -> None:
    """Combines the power consumption columns."""
    df['NE_tot'] = df['Value_NE5'] + df['Value_NE7']

def clean_time_sync(df) -> pd.DataFrame:
    """Removes the last rows with NaN values."""
    idx_drop = df[(df["NE_tot"].isna()) & (df["Date_format"] >= pd.to_datetime("2024-10-02").tz_localize('UTC'))].index
    df_time = df.drop(idx_drop)
    return df_time

def clean_humidity(df) -> None:
    """Cleans the humidity column."""
    df.loc[df['Hr [%Hr]'] > 100, 'Hr [%Hr]'] = 100

def add_corona_feature(df) -> None:
    """Adds the corona feature."""
    corona_period_1 = (df['Date_format'] >= '2020-03-18') & (df['Date_format'] <= '2020-06-06')
    corona_period_2 = (df['Date_format'] >= '2021-01-18') & (df['Date_format'] <= '2021-03-04')
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

def data_cleaning_strategy_1(df) -> pd.DataFrame:
    """NaN values are retained, and the implausible data are overwritten with NaN values."""
    df_strategy_1 = df.copy()
    df_strategy_1.loc[df_strategy_1['Value_NE5'] < 60000, 'Value_NE5'] = None
    df_strategy_1.loc[df_strategy_1['Value_NE7'] < 100000, 'Value_NE7'] = None
    combine_power_consumption(df_strategy_1)
    return df_strategy_1

def data_cleaning_strategy_2(df) -> pd.DataFrame:
    """Removes NaN values as well as the implausible values."""
    df_strategy_2 = df.copy()
    df_strategy_2 = df_strategy_2.dropna()
    df_strategy_2 = df_strategy_2.drop(df_strategy_2[df_strategy_2['Value_NE5'] < 60000].index)
    df_strategy_2 = df_strategy_2.drop(df_strategy_2[df_strategy_2['Value_NE7'] < 100000].index)
    combine_power_consumption(df_strategy_2)
    df_strategy_2.reset_index(drop=True, inplace=True)
    return df_strategy_2

def data_cleaning_strategy_3(df) -> pd.DataFrame:
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

def get_cleaned_strategy_1() -> pd.DataFrame:
    """Returns the DataFrame with the cleaning strategy 1."""
    df = read_input()
    convert_date(df)
    combine_power_consumption(df)
    df_time = clean_time_sync(df)
    clean_humidity(df_time)
    add_corona_feature(df_time)
    df_type = clean_typ(df_time)
    return data_cleaning_strategy_1(df_type)

def get_cleaned_strategy_2() -> pd.DataFrame:
    """Returns the DataFrame with the cleaning strategy 2."""
    df = read_input()
    convert_date(df)
    combine_power_consumption(df)
    df_time = clean_time_sync(df)
    clean_humidity(df_time)
    add_corona_feature(df_time)
    df_type = clean_typ(df_time)
    return data_cleaning_strategy_2(df_type)

def get_cleaned_strategy_3() -> pd.DataFrame:
    """Returns the DataFrame with the cleaning strategy 3."""
    df = read_input()
    convert_date(df)
    combine_power_consumption(df)
    df_time = clean_time_sync(df)
    clean_humidity(df_time)
    add_corona_feature(df_time)
    df_type = clean_typ(df_time)
    return data_cleaning_strategy_3(df_type)
