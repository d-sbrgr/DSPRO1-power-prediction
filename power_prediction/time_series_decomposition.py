import pandas as pd
from prophet import Prophet


def df_to_prophet(df: pd.DataFrame, key: str) -> pd.DataFrame:
    return pd.DataFrame({"ds": df["Date"], "y": df[key]})


def get_start_and_end_dates_for_key(df: pd.DataFrame, key: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df['start'] = (df[key] == 1) & (df[key].shift(1, fill_value=0) == 0)
    df['end'] = (df[key] == 1) & (df[key].shift(-1, fill_value=0) == 0)

    return df.loc[df['start'], 'Date'].reset_index(drop=True), df.loc[df['end'], 'Date'].reset_index(drop=True)


def fit_and_predict_prophet(instance: Prophet, df: pd.DataFrame) -> pd.DataFrame:
    instance.fit(df)
    future = instance.make_future_dataframe(12, freq="ME")
    return instance.predict(future)