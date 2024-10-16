import pandas as pd
from prophet import Prophet


def df_to_prophet(df: pd.DataFrame, key: str):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    df = df.tz_convert("+0100").tz_localize(None)
    df.reset_index(inplace=True)
    return pd.DataFrame({"ds": df["Date"], "y": df[key]})


def fit_and_predict_prophet(instance: Prophet, df: pd.DataFrame) -> pd.DataFrame:
    instance.fit(df)
    future = instance.make_future_dataframe(12, freq="ME")
    return instance.predict(future)