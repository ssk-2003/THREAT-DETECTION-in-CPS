import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, Input, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

class HybridCNNDBN:
    """
    Hybrid CNN-DBN model for threat detection with performance optimizations.
    """
    
    def __init__(self, input_shape, num_classes=2, cnn_filters=64, kernel_size=3, 
                 pool_size=2, dbn_layers=3, dbn_units=256, dropout_rate=0.3):
        """
        Initialize the hybrid CNN-DBN model.
        
        Args:
            input_shape: Shape of input data (features)
            num_classes: Number of output classes
            cnn_filters: Number of CNN filters
            kernel_size: Size of CNN kernel
            pool_size: Size of pooling window
            dbn_layers: Number of DBN layers
            dbn_units: Number of units per DBN layer
            dropout_rate: Dropout rate for regularization
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.cnn_filters = cnn_filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.dbn_layers = dbn_layers
        self.dbn_units = dbn_units
        self.dropout_rate = dropout_rate
        self.model = self._build_model()
        
    def _build_model(self):
        """
        Build the hybrid CNN-DBN model.
        
        Returns:
            model: Compiled Keras model
        """
        try:
            # Handle different input shapes
            if isinstance(self.input_shape, int):
                # If input_shape is a single integer (number of features)
                input_layer = Input(shape=(self.input_shape,))
                # Reshape for CNN
                reshaped = Reshape((self.input_shape, 1))(input_layer)
            elif len(self.input_shape) == 1:
                # If input_shape is a tuple with one element
                input_layer = Input(shape=(self.input_shape[0],))
                # Reshape for CNN
                reshaped = Reshape((self.input_shape[0], 1))(input_layer)
            else:
                # If input_shape already has multiple dimensions
                input_layer = Input(shape=self.input_shape)
                reshaped = input_layer
            
            # CNN part
            x = Conv1D(filters=self.cnn_filters, kernel_size=self.kernel_size, 
                      activation='relu', padding='same')(reshaped)
            x = MaxPooling1D(pool_size=self.pool_size)(x)
            x = Dropout(self.dropout_rate)(x)
            
            # Flatten CNN output
            x = Flatten()(x)
            
            # DBN part (implemented as dense layers with specific structure)
            for i in range(self.dbn_layers):
                units = self.dbn_units // (2**i) if i > 0 else self.dbn_units
                x = Dense(units, activation='relu')(x)
                x = Dropout(self.dropout_rate)(x)
            
            # Output layer
            if self.num_classes == 2:
                # Binary classification
                output_layer = Dense(1, activation='sigmoid')(x)
            else:
                # Multi-class classification
                output_layer = Dense(self.num_classes, activation='softmax')(x)
            
            # Create and compile model
            model = Model(inputs=input_layer, outputs=output_layer)
            
            # Use a lower learning rate for better stability
            optimizer = Adam(learning_rate=0.001)
            
            # Compile with appropriate loss function
            if self.num_classes == 2:
                model.compile(optimizer=optimizer, 
                             loss='binary_crossentropy',
                             metrics=['accuracy'])
            else:
                model.compile(optimizer=optimizer,
                             loss='sparse_categorical_crossentropy',
                             metrics=['accuracy'])
            
            return model
            
        except Exception as e:
            print(f"Error building model: {str(e)}")
            # Return a simple model as fallback
            return self._build_fallback_model()
    
    def _build_fallback_model(self):
        """
        Build a simple fallback model in case the main model fails.
        
        Returns:
            model: Simple compiled Keras model
        """
        model = Sequential([
            Dense(64, activation='relu', input_shape=(self.input_shape,)),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='sigmoid') if self.num_classes == 2 else Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy' if self.num_classes == 2 else 'sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def fit(self, X, y, **kwargs):
        """
        Train the model.
        
        Args:
            X: Training features
            y: Training labels
            **kwargs: Additional arguments to pass to model.fit()
            
        Returns:
            history: Training history
        """
        # Reshape input if needed
        if len(X.shape) == 2:
            X_reshaped = X.reshape(X.shape[0], X.shape[1], 1)
        else:
            X_reshaped = X
            
        # Set default values for training
        default_kwargs = {
            'batch_size': 32,
            'epochs': 20,
            'validation_split': 0.2,
            'verbose': 1
        }
        
        # Update with user-provided kwargs
        for key, value in kwargs.items():
            default_kwargs[key] = value
            
        # Add early stopping if not provided
        if 'callbacks' not in default_kwargs:
            default_kwargs['callbacks'] = [
                EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            ]
            
        # Train the model
        return self.model.fit(X_reshaped, y, **default_kwargs)
    
    def evaluate(self, X, y, verbose=0):
        """
        Evaluate the model.
        
        Args:
            X: Test features
            y: Test labels
            verbose: Verbosity mode
            
        Returns:
            loss, accuracy: Evaluation metrics
        """
        # Reshape input if needed
        if len(X.shape) == 2:
            X_reshaped = X.reshape(X.shape[0], X.shape[1], 1)
        else:
            X_reshaped = X
            
        return self.model.evaluate(X_reshaped, y, verbose=verbose)
    
    def predict(self, X, verbose=0):
        """
        Make predictions with the model.
        
        Args:
            X: Input features
            verbose: Verbosity mode
            
        Returns:
            predictions: Model predictions
        """
        # Reshape input if needed
        if len(X.shape) == 2:
            X_reshaped = X.reshape(X.shape[0], X.shape[1], 1)
        else:
            X_reshaped = X
            
        return self.model.predict(X_reshaped, verbose=verbose)
    
    def save_weights(self, filepath):
        """
        Save model weights.
        
        Args:
            filepath: Path to save weights
        """
        self.model.save_weights(filepath)
        
    def load_weights(self, filepath):
        """
        Load model weights.
        
        Args:
            filepath: Path to load weights from
        """
        self.model.load_weights(filepath)

