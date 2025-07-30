import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def load_and_merge_data(message_path, orderbook_path):
    '''
    Load message and orderbook data given specified paths.
    Merge dataframes on index.

    Args:
        message_path (str): Path to message file.
        orderbook_path (str): Path to orderbook file.

    Returns:
        df (pd.DataFrame): Merged dataframe containing both message and orderbook data.
    '''

    message_cols = ["time", "type", "orderID", "size", "price", "direction"]
    orderbook_cols = ["ask_price1", "ask_size1", "bid_price1", "bid_size1"]

    msgs = pd.read_csv(message_path, header=None, names=message_cols)
    book = pd.read_csv(orderbook_path, header=None, names=orderbook_cols)

    df = pd.concat([msgs.reset_index(drop=True), book.reset_index(drop=True)], axis=1)

    return df

def add_features(df, features):
    '''
    Add features to dataframe and drop irrelevant columns.

    Args:
        df (pd.DataFrame): Dataframe containing both message and orderbook data.
        features (List[str]): List of features to engineer.
    
    Returns:
        df (pd.DataFrame): Dataframe with feature columns.
    '''

    if "spread" in features:
        df["spread"] = df["ask_price1"] - df["bid_price1"]

    if "mid_price" in features:
        df["mid_price"] = (df["ask_price1"] + df["bid_price1"]) / 2

    if "imbalance" in features:
        df["imbalance"] = (df["ask_price1"] - df["bid_price1"])/(df["ask_price1"] + df["bid_price1"])

    if "SMA_bid_price" in features:
       df["SMA_ask_price"] = df["ask_price1"].rolling(window=10).mean()
    
    if "SMA_bid_price" in features:
        df["SMA_bid_price"] = df["bid_price1"].rolling(window=10).mean()

    # drop NA values created from moving average features
    df = df[features].dropna().copy().reset_index(drop=True)

    return df

def create_sequences(df, window_size=25, overlap=1):
    '''
    Form overlapping sequences from dataframe.

    Args:
        df (pd.DataFrame): Dataframe with feature columns.
        window_size (int): Length of windows.
        overlap (int): Number of overlapping values in consecutive windows.

    Returns:
        (np.array): Array containing sequences of length window_size.
    '''

    data = df.values
    X = []
    for i in range(0, len(data) - window_size, overlap):
        window = data[i:i+window_size]
        if len(window) == window_size:
            X.append(window)
    return np.array(X)

def scale_sequences(sequences, scaler="standard"):
    '''
    Scale sequences on window-by-window basis.

    Args:
        sequences (np.array): Array containing sequences.
        scaler (str): String indicating which scaler to use.

    Returns:
        (np.array): Scaled sequences.
    '''

    scaled = []
    for seq in sequences:
        scaler_map = {
            "standard": StandardScaler(),
            "min_max": MinMaxScaler((-1, 1))
        }
        if scaler in scaler_map:
            scaler_ = scaler_map[scaler]
        else:
            print(f"{scaler} is not a valid scaler option.")
            return
        scaled_seq = scaler_.fit_transform(seq)
        scaled.append(scaled_seq)
    return np.array(scaled)