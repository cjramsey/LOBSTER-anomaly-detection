from datetime import datetime
import os

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def train_model(model, X, callbacks, batch_size=128, epochs=25,
                test_size=0.2):
    '''
    Train model using reconstruction errors.

    Args:
        model (tf.keras..models.Model): Untrained Keras model.
        X (np.ndarray): Sequenced training dataset.
        callbacks (List[tf.keras.callbacks.Callback]): List containing callbacks.
        batch_size (int): Number of sequences in a single batch.
        epochs (int): Number of epochs to train model on.
        test_size (float): Ratio of validation set to training set.

    Returns:
        model (tf.keras.Model): Trained keras model.
        history (tf.keras.callbacks.History) Keras model training history.
    '''

    X_train, X_val = train_test_split(X, test_size=test_size, shuffle=False)
    
    history = model.fit(
        X_train, X_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, X_val),
        callbacks=callbacks
    )

    return model, history

def plot_training_loss(history, save_fig=False):
    '''
    Plot the training and validation losses against epochs.

    Args:
        history (tf.keras.callbacks.History) Keras model training history.
        save_fig (bool): Whether or not to save the plot. Defaults to False.

    Returns:
        None (NoneType)
    '''
    
    plt.plot(history.history["loss"], color="red", label="Training Loss")
    plt.plot(history.history["val_loss"], color="blue", label="Validation Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.legend()
    plt.title("Training Loss")
    plt.tight_layout()

    if save_fig:
        base_dir = os.path.dirname(__file__)
        directory = "plots"
        file_name = f"{datetime.now().strftime('%Y%m%d_%H%M')}_loss.png"
        file_path = os.path.join(base_dir, directory, file_name)

        if not os.path.exists(directory):
            os.makedirs(directory)

        plt.savefig(file_path)

    plt.show()
