import pandas as pd
import matplotlib.pyplot as plt


def convert_date(df):
    df['Date_format'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date_format'].dt.year
    df['Month'] = df['Date_format'].dt.month
    df['Day'] = df['Date_format'].dt.day
    df['Hour'] = df['Date_format'].dt.hour
    df['Weekday'] = df['Date_format'].dt.weekday + 1
    pass


def combine_power_consumption(df):
    df['Power_consumption'] = df['Value_NE5'] + df['Value_NE7']


def fill_nan_values(df):
    columns = ['Hr [%Hr]', 'T [°C]', 'WD [°]', 'WVs [m/s]', 'WVv [m/s]', 'p [hPa]', 'RainDur [min]', 'StrGlo [W/m2]']
    for column in columns:
        nan_rows = df[df[[column]].isna().any(axis=1)]
        mean_values = (df.groupby(['Month', 'Day', 'Hour'])[column].mean().to_dict())

        for index, row in nan_rows.iterrows():
            month = row['Month']
            day = row['Day']
            hour = row['Hour']
            value = mean_values.get((month, day, hour))
            df.at[index, column] = value
    pass


def print_ranges(df):
    print(f'Min/Max value for NE5 = {df['Value_NE5'].min()}, {df['Value_NE5'].max()} in kWh')
    print(f'Min/Max value for NE7 = {df['Value_NE7'].min()}, {df['Value_NE7'].max()} in kWh')
    print(f'Min/Max value for humidity = {df['Hr [%Hr]'].min()}, {df['Hr [%Hr]'].max()} in %')
    print(f'Min/Max value for rain duration = {df['RainDur [min]'].min()}, {df['RainDur [min]'].max()} in min')
    print(f'Min/Max value for global radiation = {df['StrGlo [W/m2]'].min()}, {df['StrGlo [W/m2]'].max()} in W/m2')
    print(f'Min/Max value for temperature = {df['T [°C]'].min()}, {df['T [°C]'].max()} in °C')
    print(f'Min/Max value for wind direction = {df['WD [°]'].min()}, {df['WD [°]'].max()} in °')
    print(f'Min/Max value for wind speed = {df['WVs [m/s]'].min()}, {df['WVs [m/s]'].max()} in m/s')
    print(f'Min/Max value for wind speed = {df['WVv [m/s]'].min()}, {df['WVv [m/s]'].max()} in m/s')
    print(f'Min/Max value for air pressure = {df['p [hPa]'].min()}, {df['p [hPa]'].max()} in hPa')
    pass


def plot_data(df):
    fig, axes = plt.subplots(3, 4, figsize=(15, 10))
    columns = ['Value_NE5', 'Value_NE7', 'Hr [%Hr]', 'RainDur [min]', 'StrGlo [W/m2]', 'T [°C]', 'WD [°]', 'WVs [m/s]',
               'WVv [m/s]', 'p [hPa]', 'Vacation', 'Power_consumption']
    for i, column in enumerate(columns):
        if column in df.columns:
            row = i // 4
            col = i % 4
            df[column].plot(ax=axes[row, col], title=column)

    plt.tight_layout()
    plt.show()
    pass


def clean_invalid_values(df):
    df = df.drop(df[df['Value_NE7'] < 100000].index)
    df = df.drop(df[df['Year'] == 2014].index)
    df.loc[df['Hr [%Hr]'] > 100, 'Hr [%Hr]'] = 100
    return df


df = pd.read_csv('../data/formatted_source_data.csv')
print(df.columns)

convert_date(df)
combine_power_consumption(df)
print('Are any duplicates', df.duplicated().any())
print(df.isna().any())
print_ranges(df)
print('------------------------------------------------------------')
fill_nan_values(df)
df = clean_invalid_values(df)
print(df.isna().any())
print_ranges(df)
grouped_df = df.groupby('Year')['Power_consumption'].sum().reset_index()
grouped_df['Power_consumption'] = grouped_df['Power_consumption'] / 1000000000
print(grouped_df)

plot_data(df)
