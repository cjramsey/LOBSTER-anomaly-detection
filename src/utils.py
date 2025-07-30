import json
import os
from datetime import datetime

def get_model_summary(model):
    '''
    Retrieve summary of model layers.

    Args:
        model (tf.keras.Model): Trained keras model.

    Returns:
        summary (dict): Dictionary containing layer types and number of units if applicable.
    '''
    
    summary = []
    for layer in model.layers:
        layer_type = layer.__class__.__name__
        config = layer.get_config()
        
        units = config.get("units") or config.get("filters") or config.get("pool_size") or "N/A"
        
        summary.append({
            "layer_type": layer_type,
            "units": units
        })
    return summary

def save_model_metadata(history, model, model_id, hyperparams=None, callbacks=None, log_path='models/model_history_log.json'):
    '''
    Save training history, model architecture, and metadata to a single JSON file.

    Args:
        history (tf.keras.callbacks.History): Keras model training history.
        model (tf.keras.Model): Trained Keras model.
        model_id (str): Unique identifier for the model.
        hyperparams (dict): Dictionary of hyperparameters used in training.
        callbacks (list): List of callbacks used during training (optional).
        log_path (str): Path to the JSON log file.

    Returns:
        None (NoneType)
    '''

    final_loss = history.history['loss'][-1]
    final_val_loss = history.history.get('val_loss', [None])[-1]

    callback_info = []
    if callbacks:
        for cb in callbacks:
            cb_type = type(cb).__name__
            if cb_type == "EarlyStopping":
                cb_dict = {
                    "type": cb_type,
                    "monitor": cb.monitor,
                    "patience": cb.patience,
                    "restore_best_weights": cb.restore_best_weights
                }
            elif cb_type == "ReduceLROnPlateau":
                cb_dict = {
                    "type": cb_type,
                    "monitor": cb.monitor,
                    "factor": cb.factor,
                    "patience": cb.patience
                }
            else:
                cb_dict = {"type": cb_type}
            callback_info.append(cb_dict)

    log_entry = {
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "final_loss": final_loss,
        "final_val_loss": final_val_loss,
        "full_loss": history.history['loss'],
        "full_val_loss": history.history.get('val_loss', []),
        "hyperparameters": hyperparams if hyperparams else {},
        "model_architecture": model.to_json(),
        "callbacks": callback_info
    }

    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            log_data = json.load(f)
    else:
        log_data = {}

    log_data[model_id] = log_entry

    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=4)


def save_model_with_id(model, model_id, directory="models"):
    '''
    Save keras model with unique id.

    Args:
        model (tf.keras.Model): Trained Keras model.
        model_id (str): Unique identifier for the model.
        directory (str): Folder to save model to.

    Returns:
        None (NoneType).
    '''

    if not os.path.exists(directory):
        os.makedirs(directory)

    model_path = os.path.join(directory, f"model_{model_id}.keras")
    model.save(model_path)
