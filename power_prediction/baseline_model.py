import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def split_data_train_test(df):
    """Splits data into train and test sets. Test set contains all values for the last year"""
    train_df = df.iloc[:-8670]
    test_df = df.tail(8670)
    return train_df, test_df


def plot_residuals(y_test, y_pred):
    """Plots residuals between test and prediction with different input types"""

    if isinstance(y_test, pd.Series):
        y_test_values = y_test.values.flatten()
    elif isinstance(y_test, pd.DataFrame):
        y_test_values = y_test.iloc[:, 0].values.flatten()
    elif isinstance(y_test, np.ndarray):
        y_test_values = y_test.flatten()
    else:
        raise ValueError("y_test must be a pandas Series, DataFrame, or numpy ndarray")

    if isinstance(y_pred, pd.DataFrame):
        y_pred_values = y_pred['Predictions'].values.flatten()
    elif isinstance(y_pred, pd.Series):
        y_pred_values = y_pred.values.flatten()
    elif isinstance(y_pred, np.ndarray):
        y_pred_values = y_pred.flatten()
    else:
        raise ValueError("y_pred_df must be a pandas Series, DataFrame, or numpy ndarray")

    residuals = y_test_values - y_pred_values
    plt.figure(figsize=(10,6))
    plt.scatter(range(len(residuals)), residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Residual Plot')
    plt.xlabel('Index')
    plt.ylabel('Residuals')
