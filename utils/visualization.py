import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc

def plot_confusion_matrix(y_true, y_pred):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        fig: Matplotlib figure
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    
    # Set labels
    classes = ['Normal', 'Threat']
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes[:cm.shape[1]],
           yticklabels=classes[:cm.shape[0]],
           title='Confusion Matrix',
           ylabel='True label',
           xlabel='Predicted label')
    
    # Rotate tick labels and set alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    fig.tight_layout()
    return fig

def plot_roc_curve(y_true, y_prob):
    """
    Plot ROC curve.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        
    Returns:
        fig: Matplotlib figure
    """
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc="lower right")
    
    return fig

def plot_performance_metrics(metrics):
    """
    Plot performance metrics as a radar chart.
    
    Args:
        metrics: Dictionary containing metrics
        
    Returns:
        fig: Matplotlib figure
    """
    # Extract metrics
    metrics_values = [
        metrics['sensitivity'],
        metrics['accuracy'],
        metrics['precision'],
        metrics['specificity']
    ]
    
    # Define metrics labels
    metrics_labels = [
        'Sensitivity',
        'Accuracy',
        'Precision',
        'Specificity'
    ]
    
    # Number of metrics
    num_metrics = len(metrics_values)
    
    # Calculate angles for radar chart
    angles = np.linspace(0, 2*np.pi, num_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    # Add values to close the loop
    metrics_values += metrics_values[:1]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(polar=True))
    
    # Plot metrics
    ax.plot(angles, metrics_values, 'o-', linewidth=2)
    ax.fill(angles, metrics_values, alpha=0.25)
    
    # Set labels
    ax.set_thetagrids(np.degrees(angles[:-1]), metrics_labels)
    
    # Set y-axis limits
    ax.set_ylim(0, 1)
    
    # Add title
    ax.set_title('Performance Metrics', size=14)
    
    # Add grid
    ax.grid(True)
    
    return fig

def plot_training_history(history):
    """
    Plot training history.
    
    Args:
        history: Training history dictionary
        
    Returns:
        fig: Matplotlib figure
    """
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot accuracy
    if 'accuracy' in history and 'val_accuracy' in history:
        ax1.plot(history['accuracy'], label='Training Accuracy')
        ax1.plot(history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
    
    # Plot loss
    if 'loss' in history and 'val_loss' in history:
        ax2.plot(history['loss'], label='Training Loss')
        ax2.plot(history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
    
    fig.tight_layout()
    return fig