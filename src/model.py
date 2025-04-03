import tensorflow as tf
import numpy as np
import joblib
import os
import datetime
import logging
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Set up logger
logger = logging.getLogger(__name__)

# Define models directory
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')

def create_neural_network_model(input_shape, num_classes):
    """
    Create a neural network model with the architecture from the original code
    
    Args:
        input_shape (int): Number of input features
        num_classes (int): Number of output classes
    
    Returns:
        keras.Model: Compiled Keras model
    """
    model = Sequential()
    
    model.add(Dense(256, activation='relu', input_shape=(input_shape,)))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    
    model.add(Dense(32, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(optimizer=RMSprop(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

def train_model(X_train, Y_train, X_val, Y_val, epochs=50, batch_size=32):
    """
    Train a new model from scratch and save it
    
    Args:
        X_train (np.array): Training features
        Y_train (np.array): Training labels
        X_val (np.array): Validation features
        Y_val (np.array): Validation labels
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
    
    Returns:
        tuple: Trained model and training history
    """
    # Ensure output directory exists
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Create model
    input_shape = X_train.shape[1]
    num_classes = len(np.unique(Y_train))
    model = create_neural_network_model(input_shape, num_classes)
    
    # Setup callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=os.path.join(MODELS_DIR, 'model_rmsprop.h5'),
            save_best_only=True,
            monitor='val_loss',
            mode='min',
            verbose=1
        )
    ]
    
    # Train model
    history = model.fit(
        X_train, Y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, Y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # Save the trained model in multiple formats
    model.save(os.path.join(MODELS_DIR, 'model_rmsprop.h5'))
    joblib.dump(model, os.path.join(MODELS_DIR, 'model_rmsprop.pkl'))
    
    return model, history

def build_new_model(input_shape, num_classes=3):
    """
    Build a new model with the specified input shape
    
    Args:
        input_shape (int): Number of input features
        num_classes (int): Number of output classes
    
    Returns:
        tf.keras.Model: New model
    """
    logger.info(f"Building new model with input shape: {input_shape}")
    
    # Use the same architecture as create_neural_network_model for consistency
    model = Sequential()
    
    model.add(Dense(256, activation='relu', input_shape=(input_shape,)))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    
    model.add(Dense(32, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(
        optimizer=RMSprop(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def fine_tune_model(
    model,
    X_train,
    Y_train,
    X_val,
    Y_val,
    is_new_model=False,  # Added this parameter
    batch_size=32,
    epochs=20,
    patience=5,
    learning_rate=0.0001
):
    """
    Fine-tune an existing model or train a new model

    Parameters
    ----------
    model : tensorflow.keras.Model
        Existing model to be fine-tuned
    X_train : array-like
        Training feature dataset.
    Y_train : array-like
        Training label dataset.
    X_val : array-like
        Validation feature dataset.
    Y_val : array-like
        Validation label dataset.
    is_new_model : bool
        Whether this is a new model or fine-tuning existing
    batch_size : int, optional
        Training batch size
    epochs : int, optional
        Maximum number of training epochs
    patience : int
        Early stopping patience.
    learning_rate : float
        Learning rate for fine-tuning.

    Returns
    -------
    tuple
        A tuple containing:
        - Fine-tuned model
        - Training history
    """
    # Print model summary for debugging
    model.summary()
    
    # Handle one-hot encoded labels if needed
    if len(Y_train.shape) > 1 and Y_train.shape[1] > 1:
        Y_train = np.argmax(Y_train, axis=1)
    if len(Y_val.shape) > 1 and Y_val.shape[1] > 1:
        Y_val = np.argmax(Y_val, axis=1)

    # Ensure labels are integers
    Y_train = Y_train.astype(int)
    Y_val = Y_val.astype(int)

    # Recompile the model with low learning rate
    model.compile(
        optimizer=RMSprop(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Setup callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )

    # Fine-tuning process
    history = model.fit(
        X_train,
        Y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, Y_val),
        callbacks=[early_stopping],
        verbose=1
    )

    # Evaluate model performance
    val_loss, val_accuracy = model.evaluate(X_val, Y_val, verbose=0)
    logger.info(f"Validation Loss: {val_loss:.4f}")
    logger.info(f"Validation Accuracy: {val_accuracy:.4f}")

    # Save the model with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"cardio_model_{timestamp}.h5"
    model_path = os.path.join(MODELS_DIR, model_filename)
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    model.save(model_path)
    logger.info(f"Model saved to {model_path}")

    return model, history

def load_latest_model():
    """
    Load the specific model_rmsprop.pkl from MODELS_DIR for retraining and prediction
    
    Returns:
        keras.Model: Loaded model
    """
    model_path = os.path.join(MODELS_DIR, 'model_rmsprop.pkl')
    
    if not os.path.exists(model_path):
        logger.error(f"Model file not found at {model_path}")
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    try:
        # Load the model using joblib since it's a .pkl file
        model = joblib.load(model_path)
        logger.info(f"Loaded model from: {model_path}")
        
        # Recompile the model to ensure metrics are built
        model.compile(
            optimizer=RMSprop(learning_rate=0.001),  # Match original optimizer
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        logger.info(f"Model recompiled successfully")
        return model
    
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {e}")
        raise FileNotFoundError(f"Error loading model from {model_path}: {str(e)}")

