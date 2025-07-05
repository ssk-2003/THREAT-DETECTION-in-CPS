import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def calculate_metrics(y_true, y_pred, y_prob=None):
    """
    Calculate evaluation metrics.
    
    Args:
        y_true (ndarray): True labels
        y_pred (ndarray): Predicted labels
        y_prob (ndarray, optional): Predicted probabilities
        
    Returns:
        metrics (dict): Dictionary of evaluation metrics
    """
    # Calculate basic metrics
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'sensitivity': recall_score(y_true, y_pred, average='weighted', zero_division=0),  # Sensitivity is the same as recall
        'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }
    
    # Calculate specificity (true negative rate)
    # For binary classification
    if len(np.unique(y_true)) == 2:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    else:
        # For multi-class, calculate macro-averaged specificity
        cm = confusion_matrix(y_true, y_pred)
        specificities = []
        
        for i in range(len(cm)):
            # True negatives are all elements except those in row i and column i
            tn = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
            fp = np.sum(cm[:, i]) - cm[i, i]
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            specificities.append(specificity)
        
        metrics['specificity'] = np.mean(specificities)
    
    # Calculate AUC if probabilities are provided
    if y_prob is not None:
        try:
            metrics['auc'] = roc_auc_score(y_true, y_prob)
        except:
            metrics['auc'] = 0.5  # Default value if AUC cannot be calculated
    
    return metrics

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance.
    
    Args:
        model: Trained model
        X_test (ndarray): Test features
        y_test (ndarray): Test labels
        
    Returns:
        metrics (dict): Dictionary of evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Convert probabilities to class labels for binary classification
    if len(y_pred.shape) > 1 and y_pred.shape[1] == 1:
        y_pred_classes = (y_pred > 0.5).astype(int).flatten()
        y_prob = y_pred.flatten()
    elif len(y_pred.shape) > 1:
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_prob = np.max(y_pred, axis=1)
    else:
        y_pred_classes = (y_pred > 0.5).astype(int)
        y_prob = y_pred
    
    # Calculate metrics
    return calculate_metrics(y_test, y_pred_classes, y_prob)

def cross_validate(model, X, y, cv=5):
    """
    Perform cross-validation.
    
    Args:
        model: Model to evaluate
        X (ndarray): Features
        y (ndarray): Labels
        cv (int): Number of cross-validation folds
        
    Returns:
        results (dict): Dictionary of cross-validation results
    """
    from sklearn.model_selection import KFold
    
    # Initialize K-Fold cross-validation
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Initialize lists to store results
    accuracy_scores = []
    precision_scores = []
    sensitivity_scores = []
    specificity_scores = []
    f1_scores = []
    
    # Perform cross-validation
    for train_index, test_index in kf.split(X):
        # Split data
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Train model
        model.fit(X_train, y_train)
        
        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test)
        
        # Store results
        accuracy_scores.append(metrics['accuracy'])
        precision_scores.append(metrics['precision'])
        sensitivity_scores.append(metrics['sensitivity'])
        specificity_scores.append(metrics['specificity'])
        f1_scores.append(metrics['f1'])
    
    # Calculate mean and standard deviation
    results = {
        'accuracy': {
            'mean': np.mean(accuracy_scores),
            'std': np.std(accuracy_scores)
        },
        'precision': {
            'mean': np.mean(precision_scores),
            'std': np.std(precision_scores)
        },
        'sensitivity': {
            'mean': np.mean(sensitivity_scores),
            'std': np.std(sensitivity_scores)
        },
        'specificity': {
            'mean': np.mean(specificity_scores),
            'std': np.std(specificity_scores)
        },
        'f1': {
            'mean': np.mean(f1_scores),
            'std': np.std(f1_scores)
        }
    }
    
    return results

