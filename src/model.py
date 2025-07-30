import tensorflow as tf

def build_lstm_autoencoder(input_dim, window_size=25, latent_dim=32):
    '''
    Build LSTM autoencoder model using keras Sequential API.
    Adapt the model by changing the layers directly.

    Args:
        input_dim (int): Number of features.
        window_size (int): Length of windows.
        latent_dim (int): Lower dimension for middle layer.

    Returns:
        model (tf.keras.Model): Untrained keras model.
    '''

    model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(window_size, input_dim)),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.LSTM(latent_dim),
    tf.keras.layers.RepeatVector(window_size),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(input_dim))
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    return model
