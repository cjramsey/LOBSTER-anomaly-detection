from src.config import FEATURES, WINDOW_SIZE, OVERLAP, BATCH_SIZE, EPOCHS
from src.preprocessing import load_and_merge_data, add_features, create_sequences, scale_sequences
from src.model import build_lstm_autoencoder
from src.train import train_model
from src.evaluate import get_reconstruction_errors, get_anomaly_scores, get_skew_kurtosis
from src.utils import save_model_metadata, save_model_with_id

from datetime import datetime

import tensorflow as tf

# Loading data from file paths
msg = "data/level1/AAPL_2012-06-21_34200000_57600000_message_1.csv"
orderbook = "data/level1/AAPL_2012-06-21_34200000_57600000_orderbook_1.csv"

df = load_and_merge_data(message_path=msg, orderbook_path=orderbook)
df = add_features(df, features=FEATURES)

# Split time-series data into overlapping sequences
sequences = create_sequences(df, window_size=WINDOW_SIZE, overlap=OVERLAP)
X = scale_sequences(sequences, scaler="min_max")

# Build LSTM Autoencoder
model = build_lstm_autoencoder(input_dim=len(FEATURES), window_size=WINDOW_SIZE)

# Train model
callbacks = [tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5, verbose=1)]

model, history = train_model(model, X, callbacks, batch_size=BATCH_SIZE, epochs=EPOCHS)

# Evaluate model performance
reconstruction_errors = get_reconstruction_errors(model, X)
scores = get_anomaly_scores(reconstruction_errors)

skew, kurtosis = get_skew_kurtosis(reconstruction_errors)

# Save and log model
model_id = f"model_{datetime.now().strftime('%Y%m%d_%H%M')}"

hyperparams = {
    "window_size": WINDOW_SIZE,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "learning_rate": 0.001,
    "optimizer": "adam",
    "loss_function":"mse"
}

save_model_metadata(history, model, model_id, hyperparams=hyperparams, callbacks=callbacks)
save_model_with_id
