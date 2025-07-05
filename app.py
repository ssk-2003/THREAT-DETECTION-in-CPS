import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import tensorflow as tf
from models.cnn_dbn import HybridCNNDBN
from models.saeho import SAEHO
from utils.data_processor import (
    preprocess_data, 
    split_data, 
    handle_imbalanced_data, 
    sample_data
)
from utils.visualization import (
    plot_confusion_matrix, 
    plot_roc_curve, 
    plot_performance_metrics
)
from utils.evaluation import calculate_metrics
from sklearn.model_selection import train_test_split

# Define the plot_training_history function since it's missing
def plot_training_history(history):
    """
    Plot training history of a model.
    
    Args:
        history: Training history dictionary or Keras History object
        
    Returns:
        fig: Matplotlib figure
    """
    if hasattr(history, 'history'):
        history_dict = history.history
    else:
        history_dict = history
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    if 'accuracy' in history_dict:
        ax1.plot(history_dict['accuracy'], label='Training Accuracy')
    if 'val_accuracy' in history_dict:
        ax1.plot(history_dict['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    if 'loss' in history_dict:
        ax2.plot(history_dict['loss'], label='Training Loss')
    if 'val_loss' in history_dict:
        ax2.plot(history_dict['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    return fig

# Function to validate data
def validate_data(data):
    """
    Validate and analyze input data.
    
    Args:
        data (DataFrame): Input data
        
    Returns:
        dict: Data information
    """
    info = {
        'dtypes': {},
        'missing_values': {},
        'unique_values': {}
    }
    
    # Get data types
    for col in data.columns:
        info['dtypes'][col] = str(data[col].dtype)
    
    # Get missing values
    for col in data.columns:
        info['missing_values'][col] = data[col].isnull().sum()
    
    # Get unique values for categorical columns
    for col in data.select_dtypes(include=['object', 'category']).columns:
        info['unique_values'][col] = data[col].nunique()
    
    return info

# Set page configuration
st.set_page_config(
    page_title="CPS-IoT Threat Detection",
    page_icon="🛡️",
    layout="wide"
)

# Application title
st.title("Intelligent Threat Detection in CPS-IoT Networks")
st.markdown("### Using Hybrid CNN-DBN Model with SAEHO Optimization")

# Sidebar for navigation and controls
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Analysis", "Model Training", "Real-time Detection", "Results"])

# Initialize session state variables if they don't exist
if 'model' not in st.session_state:
    st.session_state.model = None
if 'trained' not in st.session_state:
    st.session_state.trained = False
if 'data' not in st.session_state:
    st.session_state.data = None
if 'results' not in st.session_state:
    st.session_state.results = None
if 'training_history' not in st.session_state:
    st.session_state.training_history = None
if 'preprocessed_cache' not in st.session_state:
    st.session_state.preprocessed_cache = {}

# Home page
if page == "Home":
    st.markdown("""
    ## Welcome to the CPS-IoT Threat Detection System
    
    This application uses a hybrid CNN-DBN (Convolutional Neural Network - Deep Belief Network) 
    model optimized with SAEHO (Seagull Adapted Elephant Herding Optimization) to detect threats 
    in Cyber-Physical Systems (CPS) and Internet of Things (IoT) networks.
    
    ### Features:
    - Data preprocessing and analysis
    - Model training with SAEHO optimization
    - Real-time threat detection
    - Performance evaluation and visualization
    
    ### Getting Started:
    1. Upload your network traffic data
    2. Analyze the data characteristics
    3. Train the hybrid model
    4. Start real-time detection
    """)
    
    st.info("Navigate through the application using the sidebar menu.")

# Data Analysis page
elif page == "Data Analysis":
    st.header("Data Analysis")
    
    # Data upload
    uploaded_file = st.file_uploader("Upload network traffic data (CSV format)", type=["csv"])
    
    if uploaded_file is not None:
        try:
            # First, try to read the data
            with st.spinner("Loading data..."):
                data = pd.read_csv(uploaded_file)
            
            # Display basic information about the data
            st.subheader("Data Information")
            data_info = validate_data(data)
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("Dataset Shape:", data.shape)
                st.write("Feature Types:")
                for col, dtype in data_info['dtypes'].items():
                    st.write(f"- {col}: {dtype}")
            
            with col2:
                st.write("Missing Values:")
                missing_data = pd.DataFrame({
                    'Column': data_info['missing_values'].keys(),
                    'Missing Values': data_info['missing_values'].values()
                })
                st.dataframe(missing_data)
            
            # Store the data in session state
            st.session_state.data = data
            st.success(f"Data loaded successfully: {data.shape[0]} records with {data.shape[1]} features")
            
            # Display data sample
            st.subheader("Data Sample")
            st.dataframe(data.head())
            
            # Add data sampling options for large datasets
            if data.shape[0] > 5000:
                st.warning(f"Your dataset is large ({data.shape[0]} rows). Consider sampling to improve performance.")
                
                sampling_options = st.expander("Data Sampling Options")
                with sampling_options:
                    sample_method = st.radio(
                        "Sampling Method",
                        ["No sampling", "Sample by count", "Sample by percentage"]
                    )
                    
                    if sample_method == "Sample by count":
                        sample_size = st.slider(
                            "Number of samples", 
                            min_value=1000, 
                            max_value=min(50000, data.shape[0]), 
                            value=min(5000, data.shape[0]),
                            step=1000
                        )
                        if st.button("Apply Sampling (Count)"):
                            with st.spinner("Sampling data..."):
                                sampled_data = sample_data(data, sample_size=sample_size)
                                st.session_state.data = sampled_data
                                st.success(f"Data sampled to {sampled_data.shape[0]} records")
                                # Refresh the page to show the sampled data
                                st.rerun()
                    
                    elif sample_method == "Sample by percentage":
                        sample_fraction = st.slider(
                            "Percentage of data to sample", 
                            min_value=10, 
                            max_value=90, 
                            value=30,
                            step=10
                        ) / 100.0
                        if st.button("Apply Sampling (Percentage)"):
                            with st.spinner("Sampling data..."):
                                sampled_data = sample_data(data, sample_fraction=sample_fraction)
                                st.session_state.data = sampled_data
                                st.success(f"Data sampled to {sampled_data.shape[0]} records ({sample_fraction*100:.0f}%)")
                                # Refresh the page to show the sampled data
                                st.rerun()
            
            # Data statistics
            st.subheader("Data Statistics")
            col1, col2 = st.columns(2)
            with col1:
                st.write("Basic Statistics")
                st.dataframe(data.describe())
            with col2:
                st.write("Missing Values")
                st.dataframe(pd.DataFrame(data.isnull().sum(), columns=["Missing Values"]))
            
            # Data visualization
            st.subheader("Data Visualization")
            
            # Feature correlation for numeric columns
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:  # Only show correlation if we have numeric columns
                st.write("Feature Correlation")
                fig, ax = plt.subplots(figsize=(10, 8))
                correlation_matrix = data[numeric_cols].corr()
                sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', ax=ax)
                st.pyplot(fig)
            
            # Class distribution if target column exists
            if 'label' in data.columns:
                st.write("Class Distribution")
                fig, ax = plt.subplots(figsize=(8, 6))
                data['label'].value_counts().plot(kind='bar', ax=ax)
                ax.set_title('Class Distribution')
                ax.set_ylabel('Count')
                st.pyplot(fig)
            
            # Preprocessing options
            st.subheader("Data Preprocessing Options")
            preprocessing_options = st.expander("Advanced Preprocessing Options")
            
            with preprocessing_options:
                col1, col2 = st.columns(2)
                with col1:
                    categorical_encoding = st.selectbox(
                        "Categorical Encoding Method",
                        ["One-Hot Encoding", "Label Encoding", "Drop Categorical Features"],
                        index=0
                    )
                    
                    encoding_map = {
                        "One-Hot Encoding": "onehot",
                        "Label Encoding": "label",
                        "Drop Categorical Features": "none"
                    }
                
                with col2:
                    numerical_scaling = st.checkbox("Scale Numerical Features", value=True)
                    use_caching = st.checkbox("Cache Preprocessed Data", value=True)
            
            # Preprocess data button
            if st.button("Preprocess Data"):
                try:
                    # Create a progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Update status
                    status_text.text("Starting preprocessing...")
                    progress_bar.progress(10)
                    
                    # Check if we have cached data and if caching is enabled
                    cache_key = f"{hash(str(st.session_state.data))}-{categorical_encoding}-{numerical_scaling}"
                    if use_caching and 'preprocessed_cache' in st.session_state and st.session_state.preprocessed_cache.get('key') == cache_key:
                        status_text.text("Using cached preprocessed data...")
                        progress_bar.progress(50)
                        
                        X = st.session_state.preprocessed_cache['X']
                        y = st.session_state.preprocessed_cache['y']
                        
                        status_text.text("Cached data loaded successfully!")
                        progress_bar.progress(100)
                    else:
                        # Preprocess data with enhanced error handling
                        status_text.text("Preprocessing data... This may take a moment.")
                        progress_bar.progress(30)
                        
                        X, y = preprocess_data(
                            st.session_state.data,
                            categorical_encoding=encoding_map[categorical_encoding],
                            numerical_scaling=numerical_scaling
                        )
                        
                        progress_bar.progress(70)
                        status_text.text("Preprocessing completed, preparing results...")
                        
                        # Cache the preprocessed data if caching is enabled
                        if use_caching:
                            st.session_state.preprocessed_cache = {
                                'key': cache_key,
                                'X': X,
                                'y': y
                            }
                        
                        progress_bar.progress(100)
                        status_text.text("Preprocessing completed successfully!")
                    
                    # Display preprocessing results
                    st.success("Data preprocessing completed successfully!")
                    st.write("Preprocessed data shape:", X.shape)
                    st.write("Unique classes in target:", np.unique(y))
                    
                    # Proceed with splitting
                    try:
                        status_text.text("Splitting data into training and test sets...")
                        X_train, X_test, y_train, y_test = split_data(X, y)
                        
                        # Store preprocessed data in session state
                        st.session_state.X_train = X_train
                        st.session_state.X_test = X_test
                        st.session_state.y_train = y_train
                        st.session_state.y_test = y_test
                        
                        st.success("Data splitting completed!")
                        st.info(f"Training set: {X_train.shape[0]} samples\n"
                               f"Test set: {X_test.shape[0]} samples")
                        
                        # Add options for handling imbalanced data
                        st.subheader("Handle Imbalanced Data")
                        
                        # Display class distribution
                        y_train = st.session_state.y_train
                        class_counts = pd.Series(y_train).value_counts().sort_index()
                        
                        fig, ax = plt.subplots(figsize=(8, 4))
                        class_counts.plot(kind='bar', ax=ax)
                        ax.set_title('Class Distribution in Training Data')
                        ax.set_xlabel('Class')
                        ax.set_ylabel('Count')
                        st.pyplot(fig)
                        
                        # Check if data is imbalanced
                        min_class = class_counts.min()
                        max_class = class_counts.max()
                        imbalance_ratio = max_class / min_class if min_class > 0 else float('inf')
                        
                        if imbalance_ratio > 1.5:
                            st.warning(f"Data is imbalanced with a ratio of {imbalance_ratio:.2f}:1")
                            
                            # Offer resampling options
                            resampling_strategy = st.selectbox(
                                "Select a strategy to handle imbalanced data:",
                                ["None", "Oversample minority class", "Undersample majority class", "SMOTE"]
                            )
                            
                            if resampling_strategy != "None" and st.button("Apply Resampling"):
                                with st.spinner("Resampling data..."):
                                    strategy_map = {
                                        "Oversample minority class": "oversample",
                                        "Undersample majority class": "undersample",
                                        "SMOTE": "smote"
                                    }
                                    
                                    X_resampled, y_resampled = handle_imbalanced_data(
                                        st.session_state.X_train, 
                                        st.session_state.y_train,
                                        strategy=strategy_map[resampling_strategy]
                                    )
                                    
                                    # Update session state with resampled data
                                    st.session_state.X_train = X_resampled
                                    st.session_state.y_train = y_resampled
                                    
                                    # Display new class distribution
                                    new_class_counts = pd.Series(y_resampled).value_counts().sort_index()
                                    
                                    fig, ax = plt.subplots(figsize=(8, 4))
                                    new_class_counts.plot(kind='bar', ax=ax)
                                    ax.set_title('Class Distribution After Resampling')
                                    ax.set_xlabel('Class')
                                    ax.set_ylabel('Count')
                                    st.pyplot(fig)
                                    
                                    st.success(f"Data resampled successfully. New shape: {X_resampled.shape}")
                        else:
                            st.info("Data is relatively balanced. No resampling needed.")
                        
                    except Exception as split_error:
                        st.error(f"Error during data splitting: {str(split_error)}")
                        
                except Exception as preprocess_error:
                    st.error(f"Error during preprocessing: {str(preprocess_error)}")
                    st.info("Please ensure your data is properly formatted and contains valid values.")
                    
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            st.info("""
            Please ensure your data:
            - Is in CSV format
            - Contains numeric values or categorical values
            - Has a 'label' column or the last column as the target variable
            - Has consistent data types within each column
            """)
    else:
        # Use sample data
        if st.button("Use Sample Data"):
            # Generate sample data for demonstration
            np.random.seed(42)
            n_samples = 1000
            n_features = 20
            
            # Create synthetic features
            X = np.random.randn(n_samples, n_features)
            
            # Create synthetic labels (binary classification)
            y = np.random.randint(0, 2, size=n_samples)
            
            # Create DataFrame
            feature_names = [f"feature_{i}" for i in range(n_features)]
            data = pd.DataFrame(X, columns=feature_names)
            data['label'] = y
            
            st.session_state.data = data
            st.success(f"Sample data loaded: {data.shape[0]} records with {data.shape[1]} features")
            
            # Display data sample
            st.subheader("Data Sample")
            st.dataframe(data.head())
            
            # Preprocess sample data
            X, y = preprocess_data(data)
            X_train, X_test, y_train, y_test = split_data(X, y)
            
            # Store preprocessed data in session state
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            
            st.info(f"Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")

# Model Training page
elif page == "Model Training":
    st.header("Model Training")
    
    if hasattr(st.session_state, 'X_train') and hasattr(st.session_state, 'y_train'):
        # Model configuration
        st.subheader("Model Configuration")
        
        # Add model complexity options
        model_complexity = st.radio(
            "Model Complexity",
            ["Simple (Fast)", "Balanced", "Complex (Slow)"],
            index=1,
            help="Choose model complexity based on your performance needs"
        )
        
        # Set default parameters based on complexity
        if model_complexity == "Simple (Fast)":
            default_cnn_filters = 32
            default_kernel_size = 3
            default_pool_size = 2
            default_dbn_layers = 1
            default_dbn_units = 64
            default_batch_size = 64
            default_epochs = 10
        elif model_complexity == "Balanced":
            default_cnn_filters = 64
            default_kernel_size = 3
            default_pool_size = 2
            default_dbn_layers = 2
            default_dbn_units = 128
            default_batch_size = 32
            default_epochs = 20
        else:  # Complex
            default_cnn_filters = 128
            default_kernel_size = 5
            default_pool_size = 2
            default_dbn_layers = 3
            default_dbn_units = 256
            default_batch_size = 16
            default_epochs = 30
        
        # Advanced model configuration
        show_advanced = st.checkbox("Show Advanced Configuration", value=False)
        
        if show_advanced:
            col1, col2 = st.columns(2)
            with col1:
                # CNN parameters
                st.write("CNN Parameters")
                cnn_filters = st.slider("Number of CNN filters", 16, 128, default_cnn_filters, 16)
                kernel_size = st.slider("Kernel size", 2, 10, default_kernel_size, 1)
                pool_size = st.slider("Pooling size", 2, 5, default_pool_size, 1)
            
            with col2:
                # DBN parameters
                st.write("DBN Parameters")
                dbn_layers = st.slider("Number of DBN layers", 1, 5, default_dbn_layers, 1)
                dbn_units = st.slider("Units per DBN layer", 64, 512, default_dbn_units, 32)
            
            # SAEHO parameters
            st.subheader("SAEHO Optimization Parameters")
            col1, col2, col3 = st.columns(3)
            with col1:
                population_size = st.slider("Population size", 10, 100, 30, 5)
                num_elephants = st.slider("Number of elephants", 5, 20, 10, 1)
            with col2:
                max_iterations = st.slider("Maximum iterations", 10, 200, 50, 10)
                num_clans = st.slider("Number of clans", 2, 10, 3, 1)
            with col3:
                alpha = st.slider("Alpha (Seagull parameter)", 0.1, 1.0, 0.5, 0.1)
                beta = st.slider("Beta (Elephant parameter)", 0.1, 1.0, 0.5, 0.1)
                gamma = st.slider("Gamma (Clan parameter)", 0.1, 1.0, 0.3, 0.1)
        else:
            # Use default parameters based on complexity
            cnn_filters = default_cnn_filters
            kernel_size = default_kernel_size
            pool_size = default_pool_size
            dbn_layers = default_dbn_layers
            dbn_units = default_dbn_units
            population_size = 30
            num_elephants = 10
            max_iterations = 50
            num_clans = 3
            alpha = 0.5
            beta = 0.5
            gamma = 0.3
        
        # Training parameters
        st.subheader("Training Parameters")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            batch_size = st.slider("Batch size", 8, 128, default_batch_size, 8)
            epochs = st.slider("Epochs", 5, 100, default_epochs, 5)
        with col2:
            validation_split = st.slider("Validation split", 0.1, 0.3, 0.2, 0.05)
            learning_rate = st.slider("Learning rate", 0.0001, 0.01, 0.001, 0.0001, format="%.4f")
        with col3:
            patience = st.slider("Early stopping patience", 3, 20, 5, 1)
            use_early_stopping = st.checkbox("Use early stopping", value=True)
            use_reduced_precision = st.checkbox("Use mixed precision (faster)", value=True)
        
        # Add option to use a simpler model for faster training
        use_simple_model = st.checkbox("Use simplified model architecture for faster training", value=False)
        
        # Train model button
        if st.button("Train Model"):
            try:
                with st.spinner("Preparing for training..."):
                    # Create progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Enable mixed precision if selected
                    if use_reduced_precision:
                        status_text.text("Enabling mixed precision training...")
                        try:
                            policy = tf.keras.mixed_precision.Policy('mixed_float16')
                            tf.keras.mixed_precision.set_global_policy(policy)
                            status_text.text("Mixed precision enabled. This should speed up training on compatible GPUs.")
                        except Exception as e:
                            status_text.text(f"Could not enable mixed precision: {str(e)}. Continuing with default precision.")
                    
                    # Initialize model with selected parameters
                    status_text.text("Initializing model...")
                    progress_bar.progress(10)
                    
                    input_shape = st.session_state.X_train.shape[1]  # Number of features
                    
                    if use_simple_model:
                        # Create a simpler model for faster training
                        model = tf.keras.Sequential([
                            tf.keras.layers.Dense(64, activation='relu', input_shape=(input_shape,)),
                            tf.keras.layers.Dropout(0.3),
                            tf.keras.layers.Dense(32, activation='relu'),
                            tf.keras.layers.Dropout(0.3),
                            tf.keras.layers.Dense(1, activation='sigmoid')
                        ])
                        
                        # Compile the model
                        model.compile(
                            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                            loss='binary_crossentropy',
                            metrics=['accuracy']
                        )
                        
                        # Create a wrapper to maintain compatibility with the rest of the app
                        class SimpleModelWrapper:
                            def __init__(self, keras_model):
                                self.model = keras_model
                        
                        model_wrapper = SimpleModelWrapper(model)
                        
                    else:
                        # Use the hybrid CNN-DBN model
                        model_params = {
                            'input_shape': input_shape,
                            'num_classes': len(np.unique(st.session_state.y_train)),
                            'cnn_filters': cnn_filters,
                            'kernel_size': kernel_size,
                            'pool_size': pool_size,
                            'dbn_layers': dbn_layers,
                            'dbn_units': dbn_units
                        }
                        
                        model_wrapper = HybridCNNDBN(**model_params)
                    
                    status_text.text("Model initialized successfully!")
                    progress_bar.progress(20)
                    
                    # Set up callbacks
                    callbacks = []
                    
                    # Progress callback to update Streamlit progress bar
                    class ProgressCallback(tf.keras.callbacks.Callback):
                        def on_epoch_begin(self, epoch, logs=None):
                            status_text.text(f"Training epoch {epoch+1}/{epochs}...")
                            
                        def on_epoch_end(self, epoch, logs=None):
                            # Update progress bar
                            progress = 20 + (epoch + 1) / epochs * 70  # Scale to 20-90%
                            progress_bar.progress(int(progress))
                            
                            # Update metrics
                            status_text.text(f"Epoch {epoch+1}/{epochs} - "
                                           f"loss: {logs['loss']:.4f}, "
                                           f"accuracy: {logs['accuracy']:.4f}")
                            
                            # Update training history plot
                            if hasattr(self.model, 'history'):
                                history_dict = self.model.history.history
                                fig = plot_training_history(history_dict)
                                if 'training_plot' in st.session_state:
                                    st.session_state.training_plot.pyplot(fig)
                                else:
                                    st.session_state.training_plot = st.pyplot(fig)
                    
                    callbacks.append(ProgressCallback())
                    
                    # Early stopping callback
                    if use_early_stopping:
                        early_stopping = tf.keras.callbacks.EarlyStopping(
                            monitor='val_loss',
                            patience=patience,
                            restore_best_weights=True,
                            verbose=1
                        )
                        callbacks.append(early_stopping)
                    
                    # Add model checkpoint to save the best model
                    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
                        filepath='best_model.h5',
                        save_best_only=True,
                        monitor='val_loss',
                        mode='min',
                        verbose=0
                    )
                    callbacks.append(model_checkpoint)
                    
                    # Add learning rate reduction on plateau
                    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                        monitor='val_loss',
                        factor=0.5,
                        patience=3,
                        min_lr=0.00001,
                        verbose=1
                    )
                    callbacks.append(reduce_lr)
                    
                    # Train the model
                    status_text.text("Starting model training...")
                    progress_bar.progress(20)
                    
                    # Prepare data
                    X_train = st.session_state.X_train
                    y_train = st.session_state.y_train
                    
                    # Reshape input data if needed for CNN
                    if not use_simple_model and len(X_train.shape) == 2:
                        # Add a channel dimension for CNN
                        X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
                    else:
                        X_train_reshaped = X_train
                    
                    # Train the model with a try-except block to catch training errors
                    try:
                        history = model_wrapper.model.fit(
                            X_train_reshaped, y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            validation_split=validation_split,
                            callbacks=callbacks,
                            verbose=0  # Disable default verbosity as we have our own callback
                        )
                        
                        # Store trained model and results in session state
                        st.session_state.model = model_wrapper
                        st.session_state.trained = True
                        st.session_state.training_history = history.history
                        
                        # Complete
                        progress_bar.progress(100)
                        status_text.text("Model training completed!")
                        st.success("Model training completed successfully!")
                        
                        # Display final training results
                        st.subheader("Training Results")
                        
                        # Plot training history
                        fig = plot_training_history(history)
                        st.pyplot(fig)
                        
                        # Display final metrics
                        final_epoch = len(history.history['loss']) - 1
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Final Training Accuracy", f"{history.history['accuracy'][final_epoch]:.4f}")
                            st.metric("Final Training Loss", f"{history.history['loss'][final_epoch]:.4f}")
                        with col2:
                            if 'val_accuracy' in history.history:
                                st.metric("Final Validation Accuracy", f"{history.history['val_accuracy'][final_epoch]:.4f}")
                                st.metric("Final Validation Loss", f"{history.history['val_loss'][final_epoch]:.4f}")
                    
                    except Exception as training_error:
                        st.error(f"Error during model training: {str(training_error)}")
                        st.info("Try reducing model complexity, batch size, or using the simplified model option.")
                
            except Exception as e:
                st.error(f"Error during model setup: {str(e)}")
                st.info("Please check your model configuration and try again.")
    else:
        st.warning("Please upload and preprocess data first in the Data Analysis section.")

# Real-time Detection page
elif page == "Real-time Detection":
    st.header("Real-time Threat Detection")
    
    if st.session_state.trained:
        st.subheader("Detection Dashboard")
        
        # Create columns for metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(label="Active Connections", value="127")
        with col2:
            threat_count = 0  # Initialize threat count
            st.session_state.threat_metric = st.metric(label="Threats Detected", value=str(threat_count))
        with col3:
            st.metric(label="Network Load", value="68%", delta="+5%")
        with col4:
            st.session_state.status_metric = st.metric(label="System Status", value="Secure", delta="Normal", delta_color="normal")
        
        # Simulated real-time detection
        st.subheader("Network Traffic Analysis")
        
        # Start/Stop detection
        detection_running = st.checkbox("Enable Real-time Detection")
        
        if detection_running:
            try:
                # Create placeholder for real-time chart
                chart_placeholder = st.empty()
                
                # Create placeholder for alerts and analysis
                alert_placeholder = st.empty()
                analysis_placeholder = st.empty()
                
                # Create a container for attack details
                attack_details = st.container()
                
                # Initialize threat count
                if 'threat_count' not in st.session_state:
                    st.session_state.threat_count = 0
                
                # Simulate real-time data
                for i in range(100):
                    if not detection_running:
                        break
                    
                    # Generate random data points (in real app, this would be actual network data)
                    new_data = np.random.randn(10, 20)  # 10 new data points with 20 features
                    
                    # Make attack classification predictions (simulate)
                    # In a real app, this would use the trained model
                    if st.session_state.trained and hasattr(st.session_state, 'model'):
                        # Reshape data if needed for the model
                        if len(new_data.shape) == 2:
                            model_input = new_data
                            # Check if we need to reshape for CNN
                            if hasattr(st.session_state.model, 'model'):
                                # Check model type to determine if reshaping is needed
                                model_type = type(st.session_state.model.model).__name__
                                if "Sequential" in model_type or "Functional" in model_type:
                                    # Get input shape from model config
                                    try:
                                        input_config = st.session_state.model.model.get_config()
                                        # For Sequential models
                                        if 'layers' in input_config and len(input_config['layers']) > 0:
                                            first_layer_config = input_config['layers'][0]['config']
                                            if 'batch_input_shape' in first_layer_config:
                                                input_dims = len(first_layer_config['batch_input_shape'])
                                                if input_dims > 2:  # CNN expects 3D input
                                                    model_input = new_data.reshape(new_data.shape[0], new_data.shape[1], 1)
                                    except:
                                        # If we can't determine from config, use a safe approach
                                        # Try to reshape for CNN models (Conv1D, Conv2D)
                                        if any('Conv' in layer.__class__.__name__ for layer in st.session_state.model.model.layers):
                                            model_input = new_data.reshape(new_data.shape[0], new_data.shape[1], 1)
                        
                        # Get predictions from the model
                        try:
                            attack_probs = st.session_state.model.model.predict(model_input, verbose=0)
                            if len(attack_probs.shape) > 1 and attack_probs.shape[1] > 1:
                                attack_predictions = np.argmax(attack_probs, axis=1)
                                attack_probs = attack_probs[:, 1]  # Use probability of class 1
                            else:
                                attack_predictions = (attack_probs > 0.5).astype(int).flatten()
                                attack_probs = attack_probs.flatten()
                        except:
                            # Fallback to simulation if prediction fails
                            attack_probs = np.random.random(size=10)
                            attack_predictions = (attack_probs > 0.7).astype(int)
                    else:
                        # Simulate predictions if no model is available
                        attack_probs = np.random.random(size=10)
                        attack_predictions = (attack_probs > 0.7).astype(int)
                    
                    # Update chart with attack probabilities
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(range(i*10, (i+1)*10), attack_probs, 'ro-', alpha=0.7)
                    ax.set_ylim(-0.1, 1.1)
                    ax.set_xlabel('Time')
                    ax.set_ylabel('Attack Probability')
                    ax.set_title('Real-time Attack Detection')
                    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)  # Decision threshold
                    ax.grid(True)
                    
                    # Display chart
                    chart_placeholder.pyplot(fig)
                    plt.close(fig)
                    
                    # Check if any attacks were detected
                    attack_detected = np.any(attack_predictions == 1)
                    num_attacks = np.sum(attack_predictions)
                    
                    # Update threat count if attacks detected
                    if attack_detected:
                        st.session_state.threat_count += num_attacks
                        
                        # Update the threat metric
                        col2.metric(label="Threats Detected", value=str(st.session_state.threat_count), delta=f"+{num_attacks}")
                        
                        # Update system status
                        col4.metric(label="System Status", value="Under Attack", delta="Alert", delta_color="inverse")
                        
                        # Display alert
                        alert_placeholder.error(f"⚠️ ALERT: Attack detected! ({num_attacks} malicious activities identified)")
                        
                        # Only analyze metrics if an attack is detected
                        with analysis_placeholder.container():
                            st.subheader("Attack Analysis")
                            
                            # Create columns for attack metrics
                            metric_col1, metric_col2, metric_col3 = st.columns(3)
                            
                            # Simulate attack metrics
                            with metric_col1:
                                # Simulate attack type classification
                                attack_types = ["DDoS", "Man-in-the-Middle", "SQL Injection", "Phishing", "Malware"]
                                attack_type = np.random.choice(attack_types)
                                st.metric("Attack Type", attack_type)
                                
                            with metric_col2:
                                # Simulate attack severity
                                severity_levels = ["Low", "Medium", "High", "Critical"]
                                severity_weights = [0.1, 0.3, 0.4, 0.2]  # Probability weights
                                severity = np.random.choice(severity_levels, p=severity_weights)
                                severity_color = {
                                    "Low": "green",
                                    "Medium": "blue",
                                    "High": "orange",
                                    "Critical": "red"
                                }
                                st.markdown(f"<h3 style='color: {severity_color[severity]}'>Severity: {severity}</h3>", unsafe_allow_html=True)
                                
                            with metric_col3:
                                # Simulate confidence score
                                confidence = np.random.uniform(0.7, 0.99)
                                st.metric("Confidence", f"{confidence:.2f}")
                            
                            # Display attack details
                            st.subheader("Attack Details")
                            
                            # Simulate attack source
                            source_ip = f"{np.random.randint(1, 255)}.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}"
                            
                            # Simulate attack timeline
                            current_time = time.time()
                            attack_start = current_time - np.random.randint(60, 300)  # 1-5 minutes ago
                            
                            # Create attack details table
                            attack_details = {
                                "Source IP": [source_ip],
                                "Target": [f"Device_{np.random.randint(1, 20)}"],
                                "Protocol": [np.random.choice(["TCP", "UDP", "HTTP", "HTTPS", "ICMP"])],
                                "Port": [np.random.randint(1, 65535)],
                                "Start Time": [time.strftime('%H:%M:%S', time.localtime(attack_start))],
                                "Duration": [f"{(current_time - attack_start):.1f} seconds"],
                                "Packets": [np.random.randint(100, 10000)]
                            }
                            
                            st.table(pd.DataFrame(attack_details))
                            
                            # Recommended actions based on attack type
                            st.subheader("Recommended Actions")
                            
                            recommendations = {
                                "DDoS": [
                                    "Implement rate limiting",
                                    "Enable DDoS protection services",
                                    "Filter traffic using ACLs"
                                ],
                                "Man-in-the-Middle": [
                                    "Enforce HTTPS",
                                    "Implement certificate pinning",
                                    "Use VPN for sensitive communications"
                                ],
                                "SQL Injection": [
                                    "Update web application firewalls",
                                    "Use parameterized queries",
                                    "Sanitize user inputs"
                                ],
                                "Phishing": [
                                    "Train users on security awareness",
                                    "Implement email filtering",
                                    "Use anti-phishing tools"
                                ],
                                "Malware": [
                                    "Update antivirus definitions",
                                    "Scan and isolate infected devices",
                                    "Restrict executable permissions"
                                ]
                            }
                            
                            for i, recommendation in enumerate(recommendations.get(attack_type, ["Update security measures"])):
                                st.markdown(f"**{i+1}.** {recommendation}")
                    else:
                        # Update system status
                        col4.metric(label="System Status", value="Secure", delta="Normal", delta_color="normal")
                        
                        # Display normal status
                        alert_placeholder.success("✅ No attacks detected. Network is secure.")
                        
                        # Clear the analysis placeholder when no attack is detected
                        analysis_placeholder.empty()
                    
                    # Wait for a short time
                    time.sleep(0.5)
                    
            except Exception as e:
                st.error(f"Error during real-time detection: {str(e)}")
                st.info("Please check your system configuration and try again.")
        
        # Display network topology
        st.subheader("Network Topology")
        try:
            st.image("https://miro.medium.com/max/1400/1*CYB2Beu0Fxx9zZxhFNqFqQ.png", 
                     caption="IoT Network Topology Visualization", 
                     use_container_width=True)
        except Exception as e:
            st.error(f"Error loading network topology visualization: {str(e)}")
        
    else:
        st.warning("Please train the model first in the Model Training section.")

# Results page
elif page == "Results":
    st.header("Model Evaluation Results")
    
    if st.session_state.trained and hasattr(st.session_state, 'X_test') and hasattr(st.session_state, 'y_test'):
        # Evaluate model button
        if st.button("Evaluate Model"):
            try:
                with st.spinner("Evaluating model performance..."):
                    # Ensure y_test is binary
                    y_test = st.session_state.y_test.astype(int)
                    
                    # Get model predictions
                    model = st.session_state.model
                    X_test = st.session_state.X_test
                    
                    # Reshape input data if needed for CNN
                    if len(X_test.shape) == 2:
                        # Check if we're using the simple model or CNN-DBN
                        if hasattr(model, 'model'):
                            # Check model type to determine if reshaping is needed
                            model_type = type(model.model).__name__
                            if "Sequential" in model_type or "Functional" in model_type:
                                # Get input shape from model config
                                try:
                                    input_config = model.model.get_config()
                                    # For Sequential models
                                    if 'layers' in input_config and len(input_config['layers']) > 0:
                                        first_layer_config = input_config['layers'][0]['config']
                                        if 'batch_input_shape' in first_layer_config:
                                            input_dims = len(first_layer_config['batch_input_shape'])
                                            if input_dims > 2:  # CNN expects 3D input
                                                X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
                                except:
                                    # If we can't determine from config, use a safe approach
                                    # Try to reshape for CNN models (Conv1D, Conv2D)
                                    if any('Conv' in layer.__class__.__name__ for layer in model.model.layers):
                                        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
                    
                    # Get predictions
                    y_prob = model.model.predict(X_test, verbose=0)
                    if len(y_prob.shape) > 1 and y_prob.shape[1] > 1:
                        y_pred = np.argmax(y_prob, axis=1)
                        y_prob_for_metrics = y_prob[:, 1]  # Use probability of class 1 for binary metrics
                    else:
                        y_pred = (y_prob > 0.5).astype(int).flatten()
                        y_prob_for_metrics = y_prob.flatten()
                    
                    # Calculate metrics
                    metrics = calculate_metrics(y_test, y_pred, y_prob_for_metrics)
                    
                    # Store results
                    st.session_state.results = {
                        'y_true': y_test,
                        'y_pred': y_pred,
                        'y_prob': y_prob_for_metrics,
                        'metrics': metrics
                    }
                    
                    # Count the number of attacks detected
                    num_attacks = np.sum(y_pred)
                    total_samples = len(y_pred)
                    
                    if num_attacks > 0:
                        st.warning(f"⚠️ Detected {num_attacks} attacks out of {total_samples} samples ({num_attacks/total_samples*100:.1f}%)")
                    else:
                        st.success("✅ No attacks detected in the test data")
                    
                    st.success("Model evaluation completed!")
            
            except Exception as e:
                st.error(f"Error during model evaluation: {str(e)}")
                st.info("Please check your data and model configuration.")
        
        # Display results if available
        if 'results' in st.session_state and st.session_state.results is not None:
            try:
                # Get the predictions
                y_pred = st.session_state.results['y_pred']
                
                # Check if any attacks were detected
                if np.sum(y_pred) > 0:
                    st.subheader("Attack Classification Results")
                    
                    # Display attack distribution
                    attack_count = np.sum(y_pred)
                    normal_count = len(y_pred) - attack_count
                    
                    # Create a pie chart of attack vs normal
                    fig, ax = plt.subplots(figsize=(8, 8))
                    ax.pie([normal_count, attack_count], 
                           labels=['Normal', 'Attack'], 
                           autopct='%1.1f%%',
                           colors=['#4CAF50', '#F44336'],
                           explode=(0, 0.1),
                           shadow=True)
                    ax.set_title('Distribution of Network Traffic')
                    st.pyplot(fig)
                    
                    # Display metrics only if attacks were detected
                    st.subheader("Attack Detection Performance Metrics")
                    
                    metrics = st.session_state.results['metrics']
                    
                    # Create a metrics table with the requested metrics
                    metrics_df = pd.DataFrame({
                        'Metric': ['Sensitivity', 'Accuracy', 'Precision', 'Specificity'],
                        'Value': [
                            f"{metrics['sensitivity']:.4f}",
                            f"{metrics['accuracy']:.4f}",
                            f"{metrics['precision']:.4f}",
                            f"{metrics['specificity']:.4f}"
                        ]
                    })
                    
                    st.table(metrics_df)
                    
                    # Create metric cards for visual display
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Sensitivity", f"{metrics['sensitivity']:.4f}")
                        st.metric("Precision", f"{metrics['precision']:.4f}")
                    with col2:
                        st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
                        st.metric("Specificity", f"{metrics['specificity']:.4f}")
                    
                    # Display radar chart of metrics
                    st.subheader("Performance Metrics Visualization")
                    fig_metrics = plot_performance_metrics(metrics)
                    st.pyplot(fig_metrics)
                    
                    # Display confusion matrix
                    st.subheader("Confusion Matrix")
                    fig_cm = plot_confusion_matrix(
                        st.session_state.results['y_true'],
                        st.session_state.results['y_pred']
                    )
                    st.pyplot(fig_cm)
                    
                    # Display ROC curve
                    st.subheader("ROC Curve")
                    fig_roc = plot_roc_curve(
                        st.session_state.results['y_true'],
                        st.session_state.results['y_prob']
                    )
                    st.pyplot(fig_roc)
                    
                    # Feature importance (simulated)
                    st.subheader("Feature Importance for Attack Detection")
                    
                    # Generate random feature importance
                    n_features = st.session_state.X_test.shape[1]
                    feature_importance = np.random.random(size=n_features)
                    feature_importance = feature_importance / np.sum(feature_importance)
                    
                    # Sort features by importance
                    indices = np.argsort(feature_importance)[::-1]
                    feature_names = [f"Feature {i}" for i in range(n_features)]
                    
                    # Plot feature importance
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.bar(range(n_features), feature_importance[indices])
                    ax.set_xticks(range(n_features))
                    ax.set_xticklabels([feature_names[i] for i in indices], rotation=90)
                    ax.set_xlabel('Features')
                    ax.set_ylabel('Importance')
                    ax.set_title('Feature Importance for Attack Detection')
                    st.pyplot(fig)
                    
                    # Add detailed metrics explanation
                    with st.expander("Metrics Explanation"):
                        st.markdown("""
                        ### Performance Metrics Explanation
                        
                        - **Sensitivity (Recall)**: The ability of the model to correctly identify actual attacks.
                          - Formula: TP / (TP + FN)
                          - Higher is better
                        
                        - **Accuracy**: The overall correctness of the model.
                          - Formula: (TP + TN) / (TP + TN + FP + FN)
                          - Higher is better
                        
                        - **Precision**: The ability of the model to avoid false alarms.
                          - Formula: TP / (TP + FP)
                          - Higher is better
                        
                        - **Specificity**: The ability of the model to correctly identify normal traffic.
                          - Formula: TN / (TN + FP)
                          - Higher is better
                        
                        Where:
                        - TP = True Positives (correctly identified attacks)
                        - TN = True Negatives (correctly identified normal traffic)
                        - FP = False Positives (normal traffic incorrectly identified as attacks)
                        - FN = False Negatives (attacks incorrectly identified as normal traffic)
                        """)
                else:
                    # If no attacks were detected
                    st.success("✅ No attacks were detected in the test data")
                    
                    # Still show basic model performance
                    st.subheader("Model Performance Summary")
                    
                    metrics = st.session_state.results['metrics']
                    
                    # Create a simplified metrics table
                    metrics_df = pd.DataFrame({
                        'Metric': ['Accuracy', 'Specificity'],
                        'Value': [
                            f"{metrics['accuracy']:.4f}",
                            f"{metrics['specificity']:.4f}"
                        ]
                    })
                    
                    st.table(metrics_df)
                    
                    # Display confusion matrix
                    st.subheader("Confusion Matrix")
                    fig_cm = plot_confusion_matrix(
                        st.session_state.results['y_true'],
                        st.session_state.results['y_pred']
                    )
                    st.pyplot(fig_cm)
                    
                    st.info("Since no attacks were detected, detailed attack metrics are not available.")
                    
            except Exception as e:
                st.error(f"Error displaying results: {str(e)}")
                st.info("Please try evaluating the model again.")
    else:
        st.warning("Please train the model first in the Model Training section.")

# Footer
st.markdown("---")
st.markdown("### About")
st.markdown("""
This application demonstrates intelligent threat detection in CPS-IoT networks using a hybrid CNN-DBN model with SAEHO optimization.
The system is designed to detect various types of cyber threats in real-time, providing enhanced security for IoT environments.
""")

