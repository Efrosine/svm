import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import seaborn as sns
from pathlib import Path

# Import our custom SVM and feature extraction modules
from svm import SVM
from feature_extraction import extract_features_from_dataset, visualize_sample_processing

# Define paths
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
OUTPUT_DIR = os.path.join(DATA_DIR, 'features')

# Make sure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def train_apple_disease_svm(X, y, test_size=0.2, random_state=42):
    """
    Train an SVM model for apple disease detection
    
    Parameters:
    -----------
    X : numpy.ndarray
        Feature matrix
    y : numpy.ndarray
        Labels
    test_size : float, optional
        Proportion of the dataset to use for testing
    random_state : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    tuple
        Trained SVM model, training data, and testing data
    """
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    
    # Initialize and train SVM model
    # Using parameters suitable for this dataset
    svm = SVM(learning_rate=0.001, lambda_param=0.01, n_iterations=1000, epsilon=1e-4, C=1.0)
    
    print("\nTraining SVM model...")
    svm.fit(X_train, y_train)
    print("Training complete.")
    
    return svm, (X_train, y_train), (X_test, y_test)

def evaluate_model(svm, X_train, y_train, X_test, y_test):
    """
    Evaluate the trained SVM model
    
    Parameters:
    -----------
    svm : SVM
        Trained SVM model
    X_train : numpy.ndarray
        Training features
    y_train : numpy.ndarray
        Training labels
    X_test : numpy.ndarray
        Testing features
    y_test : numpy.ndarray
        Testing labels
        
    Returns:
    --------
    dict
        Model evaluation metrics
    """
    # Make predictions
    y_train_pred = np.array([svm.predict(x) for x in X_train])
    y_test_pred = np.array([svm.predict(x) for x in X_test])
    
    # Calculate accuracy
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print(f"\nTraining Accuracy: {train_accuracy * 100:.2f}%")
    print(f"Testing Accuracy: {test_accuracy * 100:.2f}%")
    
    # Calculate classification report
    print("\nClassification Report (Test Set):")
    report = classification_report(y_test, y_test_pred, 
                                   target_names=['Apple Scab', 'Healthy'],
                                   output_dict=True)
    print(pd.DataFrame(report).transpose())
    
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y_test, y_test_pred)
    
    return {
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'classification_report': report,
        'confusion_matrix': conf_matrix
    }

def visualize_results(svm, X, y, metrics, file_paths=None):
    """
    Visualize the SVM model results
    
    Parameters:
    -----------
    svm : SVM
        Trained SVM model
    X : numpy.ndarray
        Feature matrix
    y : numpy.ndarray
        Labels
    metrics : dict
        Model evaluation metrics
    file_paths : list, optional
        List of file paths corresponding to each data point
    
    Returns:
    --------
    None
    """
    # Create figure for visualizations
    fig = plt.figure(figsize=(18, 12))
    
    # Plot confusion matrix
    plt.subplot(2, 2, 1)
    sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                xticklabels=['Apple Scab', 'Healthy'],
                yticklabels=['Apple Scab', 'Healthy'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    # Plot loss curve
    plt.subplot(2, 2, 2)
    plt.plot(range(1, len(svm.losses) + 1), svm.losses, 'b-', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss per Epoch during SVM Training')
    plt.grid(True, alpha=0.3)
    
    # Plot feature importance if we have only a few features (for interpretability)
    if X.shape[1] <= 20:
        plt.subplot(2, 2, 3)
        feature_importance = np.abs(svm.w)
        sorted_idx = np.argsort(feature_importance)[::-1]
        plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx])
        plt.yticks(range(len(sorted_idx)), [f"Feature {i}" for i in sorted_idx])
        plt.xlabel('Feature Importance (|weight|)')
        plt.title('Feature Importance')
    else:
        # If we have too many features, show top 20
        plt.subplot(2, 2, 3)
        feature_importance = np.abs(svm.w)
        sorted_idx = np.argsort(feature_importance)[::-1][:20]
        plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx])
        plt.yticks(range(len(sorted_idx)), [f"Feature {i}" for i in sorted_idx])
        plt.xlabel('Feature Importance (|weight|)')
        plt.title('Top 20 Features by Importance')
    
    # Plot some support vectors if we have file paths
    if file_paths is not None and len(svm.support_vectors) > 0:
        plt.subplot(2, 2, 4)
        
        # Find the original data points corresponding to support vectors
        support_vector_indices = []
        for sv in svm.support_vectors:
            distances = np.sum((X - sv) ** 2, axis=1)
            sv_index = np.argmin(distances)
            if sv_index not in support_vector_indices:
                support_vector_indices.append(sv_index)
        
        # Limit to 9 support vectors for display
        support_vector_indices = support_vector_indices[:min(9, len(support_vector_indices))]
        
        if support_vector_indices:
            # Create a grid of images
            grid_size = int(np.ceil(np.sqrt(len(support_vector_indices))))
            for i, idx in enumerate(support_vector_indices):
                if i >= 9:  # Limit to 9 images maximum
                    break
                
                if i < len(support_vector_indices):
                    # Get the file path for this support vector
                    img_path = file_paths[idx]
                    
                    # Create a subplot for this image
                    plt.subplot(2, 2, 4)
                    plt.text(0.5, 0.5, "Support Vectors\n(See separate visualization)", 
                             ha='center', va='center', fontsize=12)
                    plt.axis('off')
            
            # Save support vectors visualization as a separate image
            sv_fig = plt.figure(figsize=(12, 12))
            for i, idx in enumerate(support_vector_indices):
                if i < len(support_vector_indices):
                    # Get the file path for this support vector
                    img_path = file_paths[idx]
                    
                    # Check if the file exists
                    if os.path.exists(img_path):
                        # Create a subplot for this image
                        plt.subplot(grid_size, grid_size, i+1)
                        img = plt.imread(img_path)
                        plt.imshow(img)
                        label = "Apple Scab" if y[idx] == -1 else "Healthy"
                        plt.title(f"{label} (SV)")
                        plt.axis('off')
            
            plt.tight_layout()
            sv_path = os.path.join(OUTPUT_DIR, 'support_vectors.png')
            sv_fig.savefig(sv_path)
            plt.close(sv_fig)
            print(f"\nSupport vectors visualization saved to: {sv_path}")
        else:
            plt.subplot(2, 2, 4)
            plt.text(0.5, 0.5, "No support vectors found", ha='center', va='center')
            plt.axis('off')
    else:
        plt.subplot(2, 2, 4)
        plt.text(0.5, 0.5, "No file paths provided\nor support vectors found", ha='center', va='center')
        plt.axis('off')
    
    plt.tight_layout()
    
    # Save the figure
    results_path = os.path.join(OUTPUT_DIR, 'apple_disease_svm_results.png')
    fig.savefig(results_path, dpi=300)
    plt.close(fig)
    
    print(f"\nResults visualization saved to: {results_path}")

def main():
    """
    Main function for apple disease detection using real images
    """
    print("Apple Disease Detection using SVM with image features")
    print("="*50)
    
    # Extract features from dataset (limited number of samples for demonstration)
    print("\nExtracting features from apple disease dataset...")
    X, y, file_paths = extract_features_from_dataset(num_samples=30)
    
    print(f"\nDataset: {len(y)} samples")
    print(f"Feature vector dimension: {X.shape[1]}")
    print(f"Class distribution: {np.sum(y == 1)} healthy, {np.sum(y == -1)} diseased (apple scab)")
    
    # Train SVM model
    svm, (X_train, y_train), (X_test, y_test) = train_apple_disease_svm(X, y)
    
    # Evaluate model
    metrics = evaluate_model(svm, X_train, y_train, X_test, y_test)
    
    # Visualize results
    visualize_results(svm, X, y, metrics, file_paths)
    
    # Process and visualize a sample image
    apple_scab_samples = [fp for fp, label in zip(file_paths, y) if label == -1]
    if apple_scab_samples:
        print("\nVisualizing sample processing for an apple scab image")
        sample_path = apple_scab_samples[0]
        viz_path, summary_path = visualize_sample_processing(sample_path)
        
        print(f"Sample processing visualization saved to: {viz_path}")
        print(f"Sample feature summary saved to: {summary_path}")
        
        # Load feature summary to display
        feature_summary = pd.read_csv(summary_path)
        print("\nSample Feature Summary:")
        print(feature_summary)
    
    print("\nApple disease detection using SVM complete!")

if __name__ == "__main__":
    main()