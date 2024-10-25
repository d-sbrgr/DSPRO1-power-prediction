from pathlib import Path


def get_project_root() -> Path:
    """Returns the root directory of the project."""
    try:
        project_root = Path(__file__).parent.parent
    except NameError:
        project_root = Path().resolve()
    return project_root

def split_data_train_test(df):
    """Splits data into train and test sets. Test set contains all values for the last year"""
    train_df = df.iloc[:-8670]
    test_df = df.tail(8670)
    return train_df, test_df