import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from collections import Counter

# Add a sampling function at the top of the file
def sample_data(data, sample_size=None, sample_fraction=None, random_state=42):
    """
    Sample data to reduce processing time.
    
    Args:
        data (DataFrame): Input data
        sample_size (int, optional): Number of samples to take
        sample_fraction (float, optional): Fraction of data to sample (0.0-1.0)
        random_state (int): Random seed for reproducibility
        
    Returns:
        DataFrame: Sampled data
    """
    if sample_size is None and sample_fraction is None:
        return data
    
    if sample_size is not None:
        if sample_size >= len(data):
            return data
        return data.sample(n=sample_size, random_state=random_state)
    
    if sample_fraction is not None:
        if sample_fraction >= 1.0:
            return data
        return data.sample(frac=sample_fraction, random_state=random_state)
    
    return data

def ensure_binary_labels(y):
    """
    Ensure labels are binary (0 or 1).
    
    Args:
        y (array-like): Input labels
        
    Returns:
        array: Binary labels
    """
    y = np.asarray(y)
    
    if y.dtype == bool:
        return y.astype(int)
    elif y.dtype == float:
        return (y > 0.5).astype(int)
    elif y.dtype == object:
        # Try to convert string labels to binary
        unique_labels = np.unique(y)
        if len(unique_labels) == 2:
            return (y == unique_labels[1]).astype(int)
        else:
            raise ValueError("Cannot convert labels to binary format")
    else:
        # For integer labels, ensure they are 0 or 1
        if set(np.unique(y)) <= {0, 1}:
            return y.astype(int)
        else:
            raise ValueError("Labels must be binary (0 or 1)")

def preprocess_data(data, categorical_encoding='onehot', numerical_scaling=True):
    """
    Preprocess the input data for model training with improved performance.

    Args:
        data (DataFrame): Input data
        categorical_encoding (str): Method for encoding categorical variables ('onehot', 'label', or 'none')
        numerical_scaling (bool): Whether to scale numerical features
        
    Returns:
        X (ndarray): Processed features
        y (ndarray): Processed labels
    """
    # Make a copy to avoid modifying the original data
    df = data.copy()

    # Extract features and target
    if 'label' in df.columns:
        X = df.drop('label', axis=1)
        y = df['label']
    else:
        # If no label column, assume the last column is the target
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

    # Identify column types
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

    # Handle missing values in numerical columns more efficiently
    if len(numerical_cols) > 0:
        X[numerical_cols] = X[numerical_cols].fillna(X[numerical_cols].median())

    # Process categorical features based on the selected method
    if len(categorical_cols) > 0:
        if categorical_encoding == 'onehot':
            # More efficient one-hot encoding
            for col in categorical_cols:
                # Fill missing values with the most common value
                X[col] = X[col].fillna(X[col].mode()[0])
                
                # One-hot encode with a more efficient approach for large datasets
                if X[col].nunique() < 10:  # For columns with few unique values
                    dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
                    X = pd.concat([X.drop(col, axis=1), dummies], axis=1)
                else:
                    # For columns with many unique values, consider label encoding instead
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col])
        
        elif categorical_encoding == 'label':
            # Label encoding (more memory efficient)
            from sklearn.preprocessing import LabelEncoder
            for col in categorical_cols:
                X[col] = X[col].fillna(X[col].mode()[0])
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])
        
        elif categorical_encoding == 'none':
            # Drop categorical columns
            X = X.drop(categorical_cols, axis=1)

    # Scale numerical features if requested
    if numerical_scaling and len(numerical_cols) > 0:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

    # Convert to numpy arrays for better performance with deep learning models
    X_array = X.values
    
    # Ensure binary labels
    try:
        y_binary = ensure_binary_labels(y)
    except ValueError as e:
        raise ValueError(f"Error converting labels to binary format: {str(e)}")

    return X_array, y_binary

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets.
    
    Args:
        X (ndarray): Features
        y (ndarray): Labels
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        X_train, X_test, y_train, y_test: Split data
    """
    # Check class distribution
    class_counts = Counter(y)
    min_samples_per_class = min(class_counts.values())
    
    # If any class has fewer than 2 samples, we can't use stratification
    if min_samples_per_class < 2:
        print(f"Warning: The least populated class has only {min_samples_per_class} member(s). "
              f"Stratification is not possible. Using random splitting instead.")
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=None)
    else:
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def create_sequences(X, y, seq_length=10, step=1):
    """
    Create sequences for time series analysis.
    
    Args:
        X (ndarray): Features
        y (ndarray): Labels
        seq_length (int): Length of each sequence
        step (int): Step size between sequences
        
    Returns:
        X_seq (ndarray): Sequence features
        y_seq (ndarray): Sequence labels
    """
    X_seq, y_seq = [], []
    
    for i in range(0, len(X) - seq_length, step):
        X_seq.append(X[i:i + seq_length])
        y_seq.append(y[i + seq_length])
    
    return np.array(X_seq), np.array(y_seq)

def handle_imbalanced_data(X, y, strategy='oversample'):
    """
    Handle imbalanced datasets.
    
    Args:
        X (ndarray): Features
        y (ndarray): Labels
        strategy (str): Strategy to handle imbalance ('oversample', 'undersample', or 'smote')
        
    Returns:
        X_resampled, y_resampled: Resampled data
    """
    try:
        if strategy == 'oversample':
            from imblearn.over_sampling import RandomOverSampler
            resampler = RandomOverSampler(random_state=42)
        elif strategy == 'undersample':
            from imblearn.under_sampling import RandomUnderSampler
            resampler = RandomUnderSampler(random_state=42)
        elif strategy == 'smote':
            from imblearn.over_sampling import SMOTE
            resampler = SMOTE(random_state=42)
        else:
            print(f"Unknown strategy: {strategy}. Using original data.")
            return X, y
            
        X_resampled, y_resampled = resampler.fit_resample(X, y)
        print(f"Resampled data shape: {X_resampled.shape}, Class distribution: {Counter(y_resampled)}")
        return X_resampled, y_resampled
    except ImportError:
        print("imblearn package not found. Using original data.")
        return X, y
    except Exception as e:
        print(f"Error during resampling: {str(e)}. Using original data.")
        return X, y

