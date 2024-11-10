import pandas as pd
from .util import get_project_root
from .preprocess_data import get_data_remove_nan, get_data_retain_nan, get_data_interpolate_nan


def _load_residuals_data() -> pd.DataFrame:
    df = pd.read_csv(get_project_root() / 'data' / 'time_decomposition_remainder_data.csv')
    df = _drop_residual_and_yhat_na(df)
    return df

def _drop_residual_and_yhat_na(df: pd.DataFrame) -> pd.DataFrame:
    columns = ["NE5_remainder", "NE5_yhat", "NE7_remainder", "NE7_yhat", "NETOT_remainder", "NETOT_yhat"]
    df.dropna(subset=columns, inplace=True)
    return df

def load_data_default_remove_nan() -> pd.DataFrame:
    return get_data_remove_nan()

def load_data_default_retain_nan() -> pd.DataFrame:
    return get_data_retain_nan()

def load_data_default_interpolate_nan() -> pd.DataFrame:
    return get_data_interpolate_nan()

def load_data_residuals_remove_nan() -> pd.DataFrame:
    data = _load_residuals_data()
    data["Date"] = pd.to_datetime(data["Date"])
    data.dropna(inplace=True)
    return data

def load_data_residuals_retain_nan() -> pd.DataFrame:
    return _load_residuals_data()

def resample_hourly_to_daily(df: pd.DataFrame, sum_cols: list, avg_cols: list, first_cols: list) -> pd.DataFrame:
    df_sum = df[sum_cols].resample('D').sum()
    df_avg = df[avg_cols].resample('D').mean()
    df_first = df[first_cols].resample('D').first()
    return pd.concat([df_sum, df_avg, df_first], axis=1)