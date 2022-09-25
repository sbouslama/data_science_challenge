
from datetime import timedelta
from typing import List
import pandas as pd
import numpy as np

def data_characterization(data: pd.DataFrame) -> pd.DataFrame:
    """Give statistical insight on the data in a dataframe
    where the index is the column name.

    Args:
        data: dataframe where the index is the column name.

    Returns:
        dataframe containing statistical information for each column of the dataset.
    """
    df = pd.DataFrame()
    count = []
    final_value_count = []
    nan_counts = data.isnull().sum().to_list()
    missing_vals = data.isnull().sum()
    nan_ratio = []
    missing_val_percent = 100 * missing_vals / data.shape[0]
    columns = data.columns
    data_description = data.describe()
    statistical_info = data_description.loc[data_description.index[1:]].T
    statistical_info.reset_index(inplace=True)
    statistical_info.rename(columns={"index": "Columns_name"}, inplace=True)
    # Attribute for each column the % of missing values
    for col in columns:
        for index, val in zip(missing_val_percent.index, missing_val_percent):
            if index == col:
                nan_ratio.append(val)
                continue
        # Value count
        i = 0
        value_count = data[col].value_counts()
        value_counts = []
        # Store top 5 values for each column based on their occurrence
        for val, occurrence in zip(value_count.index, value_count.values):
            if i <= 5:
                value_counts.append(str(val) + ":" + str(occurrence))
                i += 1
            else:
                break
        value_counts_string = ""
        for val in value_counts:
            value_counts_string = value_counts_string + val + " "
        final_value_count.append(value_counts_string)
        count.append(len(list(data[col].unique())))
    df["Columns_name"] = columns
    df["Type"] = data.dtypes.to_list()
    df["Nb_unique_values"] = count
    df["Nb_Nan_values"] = nan_counts
    df["%_Nan_values"] = nan_ratio
    df["Unique_values(value:count)"] = final_value_count
    df = df.merge(statistical_info, on="Columns_name", how="left")
    df.fillna("-", inplace=True)
    return df

def extract_date_features(df: pd.DataFrame, date_col: str)-> pd.DataFrame:
    """Extract data related features such as year, month, ect 

    Args:
        df (pd.DataFrame): dataframe with the date column
        date_col (str): name of the date column to transform

    Returns:
        pd.DataFrame: data frame with the new data related features
    """
    df["year"] = df[date_col].dt.year
    df["month"] = df[date_col].dt.month
    df["week"] = df[date_col].dt.week
    df["day"] = df[date_col].dt.day
    # df["hour"] = df[date_col].dt.hour
    # df["minute"] = df[date_col].dt.minute
    df["dayofweek"] = df[date_col].dt.dayofweek

    return df


def drop_highly_correlated_features(df: pd.DataFrame, except_cols: List[str], threashold: float = 0.95) -> List[str]:
    """Find index of columns with correlation greater than the threshold and 

    Args:
        df (pd.DataFrame): dataframe to be processed
        except_cols (List[str]): columns to exclude from the dropping
        threshold (float, optional):. Defaults to 0.95.

    Returns:
        List[str]: data frame with no highly correlated features
    """
    # create correlation  matrix
    corr_matrix = df.corr().abs()

    # select upper traingle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Find index of columns with correlation greater than threashold
    to_drop = [column for column in upper.columns if any(upper[column] > threashold)]
    to_drop = list(set(to_drop)-set(except_cols))
    
    df = df.drop(to_drop, axis=1)
    return df

def compute_lag_features(df: pd.DataFrame, col: str, lags: List[int]=[1,2,3]):
    
    
    def _generate_lookback(date,lag):
        return date+timedelta(days=lag)

    assert col in df.columns , f"{col} does not exist in the dataframe"
    
    tmp = df[["date","station_number", col]]
    for i in lags:
        shifted = tmp.copy()
        shifted.columns = ["date","station_number", f'Lag_{col}'+str(i)]
        shifted["date"] = shifted.apply(lambda x:_generate_lookback(x.date, i), axis=1)
        df = pd.merge(df, shifted, on=["date","station_number"], how='left')
    return df


