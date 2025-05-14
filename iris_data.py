"""
Utility functions for data processing and visualization.
These functions can be reused across different AI projects.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

def load_and_split_data(X, y, test_size=0.2, random_state=42, scale=True):
    """
    Load and preprocess data with optional scaling.
    
    Parameters:
    -----------
    X : array-like
        Feature matrix
    y : array-like
        Target vector
    test_size : float
        Proportion of dataset to include in the test split
    random_state : int
        Random seed for reproducibility
    scale : bool
        Whether to scale the features
        
    Returns:
    --------
    X_train, X_test, y_train, y_test : tuple
        Split and optionally scaled data
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    if scale:
        # Scale the features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test

def plot_confusion_matrix(y_true, y_pred, labels=None, title='Confusion Matrix'):
    """
    Plot a confusion matrix using seaborn.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    labels : list
        Label names for the classes
    title : str
        Title for the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels,
                yticklabels=labels)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def plot_learning_curves(train_sizes, train_scores, test_scores, title='Learning Curves'):
    """
    Plot learning curves to visualize model performance.
    
    Parameters:
    -----------
    train_sizes : array-like
        Training set sizes
    train_scores : array-like
        Training scores
    test_scores : array-like
        Test scores
    title : str
        Title for the plot
    """
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='Training score')
    plt.plot(train_sizes, test_mean, label='Cross-validation score')
    
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1)
    
    plt.title(title)
    plt.xlabel('Training Examples')
    plt.ylabel('Score')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

def plot_feature_importance(feature_names, importances, title='Feature Importance'):
    """
    Plot feature importance scores.
    
    Parameters:
    -----------
    feature_names : list
        Names of the features
    importances : array-like
        Importance scores for each feature
    title : str
        Title for the plot
    """
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
    plt.title(title)
    plt.tight_layout()
    plt.show()

def save_model(model, filename):
    """
    Save a trained model to disk.
    
    Parameters:
    -----------
    model : object
        Trained model to save
    filename : str
        Path to save the model
    """
    import joblib
    joblib.dump(model, filename)

def load_model(filename):
    """
    Load a saved model from disk.
    
    Parameters:
    -----------
    filename : str
        Path to the saved model
        
    Returns:
    --------
    model : object
        Loaded model
    """
    import joblib
    return joblib.load(filename) 
