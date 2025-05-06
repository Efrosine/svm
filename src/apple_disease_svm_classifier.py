import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
from datetime import datetime

# Import our feature extraction module
from apple_feature_extraction import load_dataset, normalize_features, extract_features

class AppleDiseaseClassifier:
    """
    SVM classifier for apple disease detection
    """
    def __init__(self, kernel='rbf', C=1.0, gamma='scale'):
        """
        Initialize the SVM classifier
        
        Args:
            kernel (str): Kernel type for SVM ('linear', 'poly', 'rbf', 'sigmoid')
            C (float): Regularization parameter
            gamma (str or float): Kernel coefficient for 'rbf', 'poly' and 'sigmoid'
        """
        self.clf = svm.SVC(kernel=kernel, C=C, gamma=gamma, probability=True)
        self.scaler = None
        self.feature_names = [
            "R-mean", "G-mean", "B-mean", 
            "R-std", "G-std", "B-std",
            "Contrast", "Energy", "Homogeneity", "Entropy",
            "Num Contours", "Contour Area", "Circularity", "Aspect Ratio"
        ]
    
    def train(self, features, labels):
        """
        Train the SVM classifier
        
        Args:
            features (numpy.ndarray): Feature matrix
            labels (numpy.ndarray): Label vector
            
        Returns:
            self: The trained classifier
        """
        # Normalize features
        normalized_features, self.scaler = normalize_features(features)
        
        # Train the classifier
        self.clf.fit(normalized_features, labels)
        
        return self
    
    def predict(self, features):
        """
        Make predictions using the trained classifier
        
        Args:
            features (numpy.ndarray): Feature matrix
            
        Returns:
            numpy.ndarray: Predicted labels
        """
        # Normalize features using the same scaler
        if self.scaler is not None:
            features = self.scaler.transform(features)
        
        # Make predictions
        return self.clf.predict(features)
    
    def predict_proba(self, features):
        """
        Get probability estimates
        
        Args:
            features (numpy.ndarray): Feature matrix
            
        Returns:
            numpy.ndarray: Probability estimates
        """
        # Normalize features using the same scaler
        if self.scaler is not None:
            features = self.scaler.transform(features)
        
        # Get probability estimates
        return self.clf.predict_proba(features)
    
    def evaluate(self, features, labels):
        """
        Evaluate the classifier
        
        Args:
            features (numpy.ndarray): Feature matrix
            labels (numpy.ndarray): True labels
            
        Returns:
            dict: Evaluation metrics
        """
        # Make predictions
        y_pred = self.predict(features)
        
        # Calculate metrics
        acc = accuracy_score(labels, y_pred)
        report = classification_report(labels, y_pred, output_dict=True)
        cm = confusion_matrix(labels, y_pred)
        
        return {
            'accuracy': acc,
            'classification_report': report,
            'confusion_matrix': cm
        }

def visualize_results(classifier, X_train, y_train, X_test, y_test, output_dir='data'):
    """
    Visualize the SVM classifier results
    
    Args:
        classifier (AppleDiseaseClassifier): Trained classifier
        X_train (numpy.ndarray): Training features
        y_train (numpy.ndarray): Training labels
        X_test (numpy.ndarray): Test features
        y_test (numpy.ndarray): Test labels
        output_dir (str): Directory to save output visualizations
        
    Returns:
        str: Path to the saved visualization
    """
    # Evaluate on test set
    results = classifier.evaluate(X_test, y_test)
    
    # Create visualization
    fig, axs = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot confusion matrix
    cm = results['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Other', 'Apple Scab'],
                yticklabels=['Other', 'Apple Scab'],
                ax=axs[0, 0])
    axs[0, 0].set_title('Confusion Matrix')
    axs[0, 0].set_ylabel('True Label')
    axs[0, 0].set_xlabel('Predicted Label')
    
    # Plot feature importance (for linear SVM only)
    if hasattr(classifier.clf, 'coef_'):
        # For linear SVM
        importance = np.abs(classifier.clf.coef_[0])
        # Sort features by importance
        indices = np.argsort(importance)[::-1]
        feature_names = np.array(classifier.feature_names)
        axs[0, 1].barh(range(len(indices)), importance[indices], align='center')
        axs[0, 1].yticks(range(len(indices)), feature_names[indices])
        axs[0, 1].set_title('Feature Importance (Linear SVM)')
    else:
        axs[0, 1].text(0.5, 0.5, 'Feature importance not available for non-linear SVM',
                      horizontalalignment='center', verticalalignment='center',
                      transform=axs[0, 1].transAxes)
        axs[0, 1].set_title('Feature Importance')
    
    # Plot decision function for a few samples (first 2 components using PCA)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    # Fit a new classifier on PCA components for visualization
    visualization_clf = svm.SVC(kernel=classifier.clf.kernel, C=classifier.clf.C, 
                             gamma=classifier.clf.gamma)
    visualization_clf.fit(X_train_pca, y_train)
    
    # Create a mesh grid
    h = 0.2  # Step size
    x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
    y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Predict on the mesh grid
    Z = visualization_clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary
    axs[1, 0].contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
    axs[1, 0].scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, 
                    edgecolors='k', cmap=plt.cm.coolwarm)
    axs[1, 0].set_title('Decision Boundary (PCA 2D Projection)')
    axs[1, 0].set_xlabel('PCA Component 1')
    axs[1, 0].set_ylabel('PCA Component 2')
    
    # Plot ROC curve
    from sklearn.metrics import roc_curve, auc
    if hasattr(classifier.clf, 'decision_function'):
        y_score = classifier.clf.decision_function(X_test)
    else:
        y_score = classifier.clf.predict_proba(X_test)[:, 1]
        
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    
    axs[1, 1].plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    axs[1, 1].plot([0, 1], [0, 1], 'k--')  # Random guess line
    axs[1, 1].set_xlim([0.0, 1.0])
    axs[1, 1].set_ylim([0.0, 1.05])
    axs[1, 1].set_xlabel('False Positive Rate')
    axs[1, 1].set_ylabel('True Positive Rate')
    axs[1, 1].set_title('Receiver Operating Characteristic (ROC)')
    axs[1, 1].legend(loc="lower right")
    
    # Add overall metrics as text
    plt.figtext(0.5, 0.01, f"Accuracy: {results['accuracy']:.4f}", ha="center", fontsize=12)
    
    # Save figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"svm_results_{timestamp}.png")
    plt.tight_layout()
    fig.savefig(output_path)
    
    return output_path

def train_and_evaluate(dataset_path, test_size=0.2, max_samples=None, output_dir='data'):
    """
    Train and evaluate an SVM classifier on the apple disease dataset
    
    Args:
        dataset_path (str): Path to the dataset
        test_size (float): Proportion of the dataset to include in the test split
        max_samples (int): Maximum number of samples to load per class
        output_dir (str): Directory to save outputs
        
    Returns:
        tuple: (classifier, evaluation results, visualization path)
    """
    print("Loading and extracting features from dataset...")
    features, labels, file_paths = load_dataset(dataset_path, max_samples=max_samples)
    
    print(f"Dataset loaded: {len(features)} samples, {sum(labels)} positive, {len(labels) - sum(labels)} negative")
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size, random_state=42, stratify=labels
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Create and train the classifier
    print("Training SVM classifier...")
    classifier = AppleDiseaseClassifier(kernel='rbf', C=1.0, gamma='scale')
    classifier.train(X_train, y_train)
    
    # Evaluate on training set
    print("Evaluating on training set...")
    train_results = classifier.evaluate(X_train, y_train)
    print(f"Training accuracy: {train_results['accuracy']:.4f}")
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_results = classifier.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_results['accuracy']:.4f}")
    
    # Visualize results
    print("Visualizing results...")
    viz_path = visualize_results(classifier, X_train, y_train, X_test, y_test, output_dir)
    print(f"Results visualization saved to: {viz_path}")
    
    return classifier, test_results, viz_path

def demonstrate_single_prediction(classifier, image_path):
    """
    Demonstrate prediction for a single image
    
    Args:
        classifier (AppleDiseaseClassifier): Trained classifier
        image_path (str): Path to the image file
        
    Returns:
        tuple: (prediction, probability)
    """
    # Extract features from the image
    features = extract_features(image_path)
    features = features.reshape(1, -1)  # Reshape to match expected format
    
    # Make prediction
    prediction = classifier.predict(features)[0]
    probabilities = classifier.predict_proba(features)[0]
    
    # Print results
    print(f"\nPrediction for image {image_path}:")
    print(f"Class: {'apple scab' if prediction == 1 else 'other'}")
    print(f"Probability of apple scab: {probabilities[1]:.4f}")
    print(f"Probability of other: {probabilities[0]:.4f}")
    
    return prediction, probabilities

if __name__ == "__main__":
    # Path to the dataset
    dataset_path = "data/dataset/apple_disease"
    
    # Train and evaluate with a small sample for demonstration
    print("Starting SVM classifier demonstration with a small sample...")
    classifier, results, viz_path = train_and_evaluate(dataset_path, max_samples=20)
    
    # Find a sample image for prediction demonstration
    apple_scab_dir = os.path.join(dataset_path, "apple", "apple scab")
    sample_image = os.path.join(apple_scab_dir, os.listdir(apple_scab_dir)[0])
    
    # Demonstrate single image prediction
    demonstrate_single_prediction(classifier, sample_image)
    
    print("\nDemonstration complete.")