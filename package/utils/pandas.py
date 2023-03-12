"""Utility functions for pandas operations"""

from typing import List
import numpy as np
import pandas as pd


def apply_filters(df: pd.DataFrame, filters: dict, reset_index=False):
    """
    Filters df based on given filters (key-values pairs).
    """
    X = df.copy()

    for col, values in filters.items():
        if isinstance(values, (list, tuple, np.ndarray)):
            indices = X[col].isin(list(values))
        else:
            indices = X[col] == values
        X = X[indices]

    if reset_index:
        X = X.reset_index(drop=True)

    return X


def apply_antifilters(df: pd.DataFrame, filters: dict, reset_index=False):
    """
    Filters df removing rows for given filters (key-values pairs).
    """
    X = df.copy()

    for col, values in filters.items():
        if isinstance(values, (list, tuple, np.ndarray)):
            indices = X[col].isin(list(values))
        else:
            indices = X[col] == values
        X = X[~indices]

    if reset_index:
        X = X.reset_index(drop=True)

    return X


def custom_eval(x):
    """Splits string '["a", "b", "c"]' into ["a", "b", "c"]."""
    if isinstance(x, str):
        x = x.replace('[', '')
        x = x.replace(']', '')

        x = x.split(',')
        x = [y.rstrip().lstrip() for y in x]
        return x
    else:
        return ['NA']


def split_column_into_columns(df, column):
    """
    For given df, splits `column` containing values like '["a", "b"]'
    into one-hot subcolumns like a. b with `Yes`/`No` values.
    """
    df[column] = df[column].apply(custom_eval)

    unique_values = []
    for i in range(len(df)):
        index = df.index[i]

        list_of_values = df.loc[index, column]

        for x in list_of_values:
            if (x != 'NA') and (x != ''):
                df.at[index, x] = 'Yes'
                if x not in unique_values:
                    unique_values.append(x)

    df[unique_values] = df[unique_values].fillna('No')
    df[f'any_{column}'] = df[unique_values].apply(
        lambda x: 'Yes' if 'Yes' in list(x) else 'No', axis=1
    )
    return df


def custom_read_csv(path: str, columns_to_onehot: List) -> pd.DataFrame:
    """Custom CSV reader

    Args:
        path (str): path to .csv file
        columns_to_onehot (List): list of columns to one-hotify

    Returns:
        pd.DataFrame: loaded df
    """
    df = pd.read_csv(path)
    for column in columns_to_onehot:
        df = split_column_into_columns(df, column)
    return df