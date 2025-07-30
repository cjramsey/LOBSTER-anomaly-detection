'''Configuration file for changing global constants/hyperparameters conveniently.'''

FEATURES = ["price", "ask_price1", "bid_price1", "spread",
            "imbalance", "SMA_ask_price", "SMA_bid_price"]

WINDOW_SIZE = 25
OVERLAP = 1

EPOCHS = 50
BATCH_SIZE = 128