from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def get_project_root() -> Path:
    """Returns the root directory of the project."""
    try:
        project_root = Path(__file__).parent.parent
    except NameError:
        project_root = Path().resolve()
    return project_root


def df_time_to_utc_plus_one(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    df = df.tz_convert("+0100").tz_localize(None)
    df.reset_index(inplace=True)
    return df


def split_data_train_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Splits data into train and test sets. Test set contains all values for the last year"""
    split_date = df["Date"].max() - pd.DateOffset(years=1)
    train_df = df[df["Date"] <= split_date]
    test_df = df[df["Date"] > split_date]
    return train_df, test_df


def make_column_names_model_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns leaving the brackets containing the units out"""
    df.columns = [col.split(" ")[0] for col in df.columns]
    return df


def plot_target_and_predictions(dates: pd.DatetimeIndex, target: pd.Series, prediction: pd.Series, title: str, x_label: str, y_label: str, ylim_bottom = 0) -> None:
    sns.set(style='whitegrid', palette='muted', font_scale=1.2)
    palette = ['#01BEFE', '#FF7D00', '#FFDD00', '#FF006D', '#ADFF02', '#8F00FF']
    sns.set_palette(sns.color_palette(palette))

    plt.figure(figsize=(15, 10))
    plt.plot(dates, target, label='Actual')
    plt.plot(dates, prediction, label='Predicted')
    plt.ylim(bottom = ylim_bottom)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.show()