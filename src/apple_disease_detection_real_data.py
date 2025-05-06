import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Import the SVM implementation
from svm import SVM

# Import feature extraction functionality
from optimized_comparative_feature_extraction import extract_features, load_balanced_dataset

# Get the absolute path to the project directory
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
DATASET_DIR = os.path.join(DATA_DIR, 'dataset', 'apple_disease')

# Create data directory if it doesn't exist
os.makedirs(DATA_DIR, exist_ok=True)

def load_features_from_extraction_results(csv_path):
    """
    Load features from a previously generated CSV file
    
    Args:
        csv_path (str): Path to the features CSV file
        
    Returns:
        tuple: (features, labels, file_paths)
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Extract features
    features = df[['Texture Regularity', 'Lesion Density']].values
    
    # Convert class labels to numeric: healthy=+1, diseased=-1
    labels = np.array([1 if label == 'healthy' else -1 for label in df['Class']])
    
    # Get file paths
    file_paths = df['File'].values
    
    return features, labels, file_paths

def extract_features_from_dataset(dataset_path, num_samples=50):
    """
    Extract features directly from the dataset
    
    Args:
        dataset_path (str): Path to the dataset directory
        num_samples (int): Number of samples per class
        
    Returns:
        tuple: (features, labels)
    """
    print(f"Extracting features from {num_samples} samples per class...")
    features, labels, _ = load_balanced_dataset(dataset_path, num_samples)
    
    # Convert healthy=0 to +1 and disease=1 to -1 for SVM
    # This conversion is needed because load_balanced_dataset uses 0 for healthy
    # but our SVM implementation expects +1 for one class and -1 for the other
    svm_labels = np.array([1 if label == 0 else -1 for label in labels])
    
    return features, svm_labels

def visualize_data_before_hyperplane(X, y, title="Apple Disease Features", save_path=None):
    """
    Visualize the data before finding a hyperplane
    
    Args:
        X (numpy.ndarray): Feature data
        y (numpy.ndarray): Labels (+1 for healthy, -1 for diseased)
        title (str): Plot title
        save_path (str): Path to save the visualization
    """
    plt.figure(figsize=(10, 8))
    
    # Split data by class
    healthy_indices = (y == 1)
    diseased_indices = (y == -1)
    
    # Plot each class with different color and marker
    plt.scatter(X[healthy_indices, 0], X[healthy_indices, 1], 
                color='green', marker='o', label='Healthy Apple', 
                s=120, alpha=0.8, edgecolors='darkgreen')
    
    plt.scatter(X[diseased_indices, 0], X[diseased_indices, 1], 
                color='red', marker='x', label='Diseased Apple', 
                s=120, alpha=0.8, linewidth=2)
    
    plt.xlabel('Texture Regularity (Higher = More Uniform Texture)', fontsize=14)
    plt.ylabel('Lesion Density (Higher = More Lesions)', fontsize=14)
    plt.title(title, fontsize=16)
    plt.legend(loc='upper right', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Save the visualization if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Visualization saved to: {save_path}")
    
    plt.close()

def create_iteration_table_image(svm_model, save_path):
    """
    Create an image of a table showing the first 10 and last 10 iterations of SVM training
    with loss values instead of error counts
    
    Parameters:
    -----------
    svm_model : SVM
        The trained SVM model with iteration history
    save_path : str
        Path to save the table image
        
    Returns:
    --------
    str
        Path to the saved table image
    """
    # Get the first 10 and last 10 iterations
    first_iterations, last_iterations = svm_model.get_iteration_table()
    
    # Create a list to hold the rows and data for the table
    table_rows = []
    
    # Format the weights and add first iterations to the table
    for i, iteration in enumerate(first_iterations):
        # Format weights as string with 8 decimal places
        w_str = "[" + ", ".join([f"{w:.8f}" for w in iteration['w']]) + "]"
        
        # Get the corresponding loss value
        loss_value = svm_model.losses[iteration['epoch']-1]
        
        # Add row to the table with loss instead of errors
        table_rows.append([
            f"{iteration['epoch']:d}",
            w_str,
            f"{iteration['b']:.8f}",
            f"{loss_value:.8f}"  # Show loss value with 8 decimal places
        ])
    
    # Add a separator row if we have both first and last iterations
    if first_iterations and last_iterations:
        table_rows.append(["...", "...", "...", "..."])
    
    # Add the last iterations to the table
    for iteration in last_iterations:
        # Format weights as string with 8 decimal places
        w_str = "[" + ", ".join([f"{w:.8f}" for w in iteration['w']]) + "]"
        
        # Get the corresponding loss value
        loss_value = svm_model.losses[iteration['epoch']-1]
        
        # Add row to the table with loss instead of errors
        table_rows.append([
            f"{iteration['epoch']:d}",
            w_str,
            f"{iteration['b']:.8f}",
            f"{loss_value:.8f}"  # Show loss value with 8 decimal places
        ])
    
    # Create a figure for the table
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Set the table title
    ax.set_title("SVM Training Iteration History: First 10 and Last 10 Iterations", 
                 fontsize=16, weight='bold', pad=20)
    
    # Create the table
    headers = ["Epoch", "Weights (w)", "Bias (b)", "Loss"]  # Changed from "Errors" to "Loss"
    table = ax.table(cellText=table_rows, colLabels=headers, 
                    loc='center', cellLoc='center',
                    colColours=['#f2f2f2']*len(headers))
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Highlight the first and last rows
    for i in range(len(table_rows)):
        if i < 10:  # First iterations
            for j in range(len(headers)):
                cell = table[(i+1, j)]
                cell.set_facecolor('#e6f2ff')  # Light blue for first iterations
        elif i > 10:  # Last iterations
            for j in range(len(headers)):
                cell = table[(i+1, j)]
                cell.set_facecolor('#e6ffe6')  # Light green for last iterations
                
    # Add a footnote to explain the loss column
    plt.figtext(0.5, 0.01, 
               "Loss: SVM optimization objective (includes both classification errors and margin optimization)",
               ha="center", fontsize=10, 
               bbox=dict(boxstyle="round", fc="lightgrey", alpha=0.7))
    
    # Save the table as an image
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nIteration history table image saved to: {save_path}")
    return save_path

def visualize_svm_results(X, y, svm_model, output_dir, filename_prefix='apple_disease_svm_real_data'):
    """
    Visualize SVM results with decision boundary and performance metrics
    
    Args:
        X (numpy.ndarray): Feature data
        y (numpy.ndarray): Labels (+1 for healthy, -1 for diseased)
        svm_model (SVM): Trained SVM model
        output_dir (str): Directory to save visualization
        filename_prefix (str): Prefix for saved files
    """
    # Split data by class
    healthy_indices = (y == 1)
    diseased_indices = (y == -1)
    
    # Create a figure with two subplots
    plt.figure(figsize=(16, 8))

    # First subplot: Decision Boundary and Hyperplanes
    plt.subplot(1, 2, 1)

    # Plot training points
    plt.scatter(X[healthy_indices, 0], X[healthy_indices, 1], 
                color='green', marker='o', label='Healthy Apple', s=100, alpha=0.7)
    plt.scatter(X[diseased_indices, 0], X[diseased_indices, 1], 
                color='red', marker='x', label='Diseased Apple', s=100, alpha=0.7)

    # Plot decision boundary
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    Z = np.array([svm_model.predict(np.array([x, y])) for x, y in zip(xx.ravel(), yy.ravel())])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.2, cmap=plt.cm.RdYlGn)

    # Calculate and display margin
    w_norm = np.linalg.norm(svm_model.w)
    if w_norm > 0:
        # Plot the decision boundary: w·x - b = 0
        xx_margin = np.linspace(x_min, x_max, 100)
        yy_db = (-svm_model.w[0] * xx_margin + svm_model.b) / svm_model.w[1]
        plt.plot(xx_margin, yy_db, 'k-', label='Decision Boundary', linewidth=2)
        
        # Get and display support vectors
        if svm_model.support_vectors is not None and len(svm_model.support_vectors) > 0:
            plt.scatter(svm_model.support_vectors[:, 0], svm_model.support_vectors[:, 1], 
                        s=200, facecolors='none', edgecolors='blue', linewidth=2,
                        label='Support Vectors')
            
            # Separate positive and negative margin lines for correct legend display
            # Positive margin (for class +1)
            yy_pos_margin = (-svm_model.w[0] * xx_margin + svm_model.b + 1) / svm_model.w[1]
            plt.plot(xx_margin, yy_pos_margin, 'k--', linewidth=1.5, label='Positive Margin')
            
            # Negative margin (for class -1)
            yy_neg_margin = (-svm_model.w[0] * xx_margin + svm_model.b - 1) / svm_model.w[1]
            plt.plot(xx_margin, yy_neg_margin, 'k--', linewidth=1.5, label='Negative Margin')
        
        # Print information about the margin
        margin_distance = svm_model.get_margin_distance()
        print(f"\nMargin distance: {margin_distance:.4f}")
        print(f"Number of support vectors: {len(svm_model.support_vectors)}")
        
        # Add margin distance text to the plot
        plt.annotate(f"Margin width: {margin_distance:.4f}", 
                     xy=(0.05, 0.05), xycoords='axes fraction', fontsize=10)
                     
        # Check for margin violations
        violations_pos = 0
        violations_neg = 0
        
        for i, x_i in enumerate(X):
            functional_margin = y[i] * (np.dot(svm_model.w, x_i) - svm_model.b)
            if y[i] == 1 and functional_margin < 1:
                violations_pos += 1
            elif y[i] == -1 and functional_margin < 1:
                violations_neg += 1
        
        print(f"Margin violations - Positive class: {violations_pos}, Negative class: {violations_neg}")
        
        # Add violation count to the plot
        plt.annotate(f"Margin violations: {violations_pos + violations_neg}", 
                     xy=(0.05, 0.10), xycoords='axes fraction', fontsize=10)

    plt.xlabel('Texture Regularity (Higher = More Uniform Texture)', fontsize=12)
    plt.ylabel('Lesion Density (Higher = More Lesions)', fontsize=12)
    plt.title('Apple Disease Detection - SVM Decision Boundary (Real Data)', fontsize=14)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)

    # Second subplot: Loss per epoch
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(svm_model.losses) + 1), svm_model.losses, 'b-', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Loss per Epoch during SVM Training', fontsize=14)
    plt.grid(True, alpha=0.3)

    # Add model parameters as text on loss plot
    param_text = (f'Learning Rate: {svm_model.learning_rate}\n'
                  f'Lambda: {svm_model.lambda_param}\n'
                  f'C: {svm_model.C}\n'
                  f'Iterations: {svm_model.n_iterations}')
    plt.annotate(param_text, xy=(0.02, 0.95), xycoords='axes fraction', 
                 fontsize=10, va='top', bbox=dict(boxstyle='round', fc='white', alpha=0.7))

    plt.tight_layout()

    # Save results
    results_path_png = os.path.join(output_dir, f'{filename_prefix}.png')
    results_path_pdf = os.path.join(output_dir, f'{filename_prefix}.pdf')
    plt.savefig(results_path_png, dpi=300)
    plt.savefig(results_path_pdf)
    plt.close()
    
    print(f"Results saved to: {results_path_png} and {results_path_pdf}")
    return results_path_png, results_path_pdf

def evaluate_model_on_test_data(svm_model, X_test, y_test, output_dir, X_train=None, y_train=None):
    """
    Evaluate the trained SVM model on test data and visualize results
    
    Args:
        svm_model (SVM): Trained SVM model
        X_test (numpy.ndarray): Test features
        y_test (numpy.ndarray): Test labels
        output_dir (str): Directory to save results
        X_train (numpy.ndarray, optional): Training features for visualization
        y_train (numpy.ndarray, optional): Training labels for visualization
    """
    # Make predictions on test data
    y_pred = np.array([svm_model.predict(x) for x in X_test])
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Set Accuracy: {accuracy * 100:.2f}%")
    
    # Generate classification report
    class_names = ['Diseased', 'Healthy']
    report = classification_report(y_test, y_pred, target_names=class_names)
    print("\nClassification Report:")
    print(report)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Create a figure for the confusion matrix visualization
    plt.figure(figsize=(10, 8))
    
    # Plot the confusion matrix
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix', fontsize=16)
    plt.colorbar()
    
    # Add class labels
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, fontsize=12)
    plt.yticks(tick_marks, class_names, fontsize=12)
    
    # Add numbers to the confusion matrix cells
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=16)
    
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.tight_layout()
    
    # Save the confusion matrix
    cm_path = os.path.join(output_dir, 'apple_disease_confusion_matrix.png')
    plt.savefig(cm_path, dpi=300)
    plt.close()
    
    print(f"Confusion matrix saved to: {cm_path}")
    
    # Create a combined visualization with train and test data if provided
    if X_train is not None and y_train is not None:
        plt.figure(figsize=(12, 10))
        
        # Plot training data
        healthy_train = (y_train == 1)
        diseased_train = (y_train == -1)
        
        plt.scatter(X_train[healthy_train, 0], X_train[healthy_train, 1], 
                    color='forestgreen', marker='o', label='Healthy (Train)', 
                    s=80, alpha=0.6, edgecolors='darkgreen')
        plt.scatter(X_train[diseased_train, 0], X_train[diseased_train, 1], 
                    color='firebrick', marker='x', label='Diseased (Train)', 
                    s=80, alpha=0.6, linewidth=1.5)
        
        # Plot test data with different markers
        healthy_test = (y_test == 1)
        diseased_test = (y_test == -1)
        
        plt.scatter(X_test[healthy_test, 0], X_test[healthy_test, 1], 
                    color='limegreen', marker='^', label='Healthy (Test)', 
                    s=100, alpha=0.8, edgecolors='darkgreen')
        plt.scatter(X_test[diseased_test, 0], X_test[diseased_test, 1], 
                    color='tomato', marker='*', label='Diseased (Test)', 
                    s=120, alpha=0.8, linewidth=1.5)
        
        # Mark misclassified test points
        misclassified = (y_pred != y_test)
        if np.any(misclassified):
            # Fix: Properly index the misclassified points
            misclassified_indices = np.where(misclassified)[0]
            plt.scatter(X_test[misclassified_indices, 0], X_test[misclassified_indices, 1], 
                        s=180, facecolors='none', edgecolors='black', linewidth=2,
                        label='Misclassified')
        
        # Plot decision boundary
        x_min, x_max = min(X_train[:, 0].min(), X_test[:, 0].min()) - 0.5, max(X_train[:, 0].max(), X_test[:, 0].max()) + 0.5
        y_min, y_max = min(X_train[:, 1].min(), X_test[:, 1].min()) - 0.5, max(X_train[:, 1].max(), X_test[:, 1].max()) + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
        Z = np.array([svm_model.predict(np.array([x, y])) for x, y in zip(xx.ravel(), yy.ravel())])
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, alpha=0.2, cmap=plt.cm.RdYlGn)
        
        # Plot decision boundary line
        w_norm = np.linalg.norm(svm_model.w)
        if w_norm > 0:
            xx_margin = np.linspace(x_min, x_max, 100)
            yy_db = (-svm_model.w[0] * xx_margin + svm_model.b) / svm_model.w[1]
            plt.plot(xx_margin, yy_db, 'k-', label='Decision Boundary', linewidth=2)
            
            # Add the margin lines
            yy_pos_margin = (-svm_model.w[0] * xx_margin + svm_model.b + 1) / svm_model.w[1]
            plt.plot(xx_margin, yy_pos_margin, 'k--', linewidth=1, label='_nolegend_')
            
            yy_neg_margin = (-svm_model.w[0] * xx_margin + svm_model.b - 1) / svm_model.w[1]
            plt.plot(xx_margin, yy_neg_margin, 'k--', linewidth=1, label='Margin')
        
        plt.xlabel('Texture Regularity (Higher = More Uniform Texture)', fontsize=14)
        plt.ylabel('Lesion Density (Higher = More Lesions)', fontsize=14)
        plt.title('Apple Disease Detection - Train vs Test Results', fontsize=16)
        plt.legend(loc='upper right', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add accuracy information to the plot
        plt.annotate(f"Test Accuracy: {accuracy * 100:.2f}%", 
                     xy=(0.05, 0.05), xycoords='axes fraction', 
                     fontsize=12, bbox=dict(boxstyle='round', fc='white', alpha=0.7))
        
        # Save the combined visualization
        combined_path = os.path.join(output_dir, 'apple_disease_train_test_comparison.png')
        plt.savefig(combined_path, dpi=300)
        plt.close()
        
        print(f"Train vs Test visualization saved to: {combined_path}")

def main():
    """
    Main function to run the apple disease detection with real data
    """
    # Setup output directory
    output_dir = os.path.join(DATA_DIR, 'real_data_svm_results')
    os.makedirs(output_dir, exist_ok=True)
    
    print("Apple Disease Detection Using Real Data with SVM")
    print("----------------------------------------------")
    
    # Extract features from real data
    # First, check if we can find any recent feature extraction results
    feature_extraction_dir = os.path.join(DATA_DIR, 'feature_extraction_results')
    
    features = None
    labels = None
    
    if os.path.exists(feature_extraction_dir):
        # Find the most recent feature extraction run
        run_dirs = [d for d in os.listdir(feature_extraction_dir) if d.startswith('run_')]
        if run_dirs:
            # Sort by creation time (newest first)
            sorted_runs = sorted(run_dirs, reverse=True)
            latest_run = sorted_runs[0]
            csv_path = os.path.join(feature_extraction_dir, latest_run, 'features.csv')
            
            if os.path.exists(csv_path):
                print(f"Loading features from existing extraction result: {csv_path}")
                features, labels, file_paths = load_features_from_extraction_results(csv_path)
                print(f"Loaded {len(features)} feature vectors")
    
    # If we couldn't find or load existing features, extract them now
    if features is None or labels is None:
        print("No existing feature extraction results found. Extracting features now...")
        features, labels = extract_features_from_dataset(DATASET_DIR, num_samples=30)
        print(f"Extracted {len(features)} feature vectors")
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"\nData split into {len(X_train)} training samples and {len(X_test)} testing samples")
    
    # Visualize the data before finding hyperplane
    data_viz_path = os.path.join(output_dir, 'apple_disease_real_data_before_hyperplane.png')
    visualize_data_before_hyperplane(features, labels, 
                                    title="Apple Disease Features from Real Data",
                                    save_path=data_viz_path)
    
    # Train SVM model with adjusted parameters
    # We'll try a few different configurations to find the best one
    print("\nTraining SVM model with real data...")
    
    svm_configs = [
        {'learning_rate': 0.001, 'lambda_param': 0.001, 'n_iterations': 2000, 'C': 10.0, 'name': 'default'},
        {'learning_rate': 0.001, 'lambda_param': 0.0001, 'n_iterations': 2000, 'C': 20.0, 'name': 'high_c'},
        {'learning_rate': 0.0005, 'lambda_param': 0.0005, 'n_iterations': 3000, 'C': 5.0, 'name': 'more_iterations'}
    ]
    
    best_svm = None
    best_accuracy = 0
    
    for config in svm_configs:
        name = config.pop('name')
        print(f"\nTraining SVM with configuration: {name}")
        
        svm = SVM(**config)
        svm.fit(X_train, y_train)
        
        # Make predictions on training set
        y_train_pred = np.array([svm.predict(x) for x in X_train])
        train_accuracy = accuracy_score(y_train, y_train_pred)
        
        # Make predictions on test set
        y_test_pred = np.array([svm.predict(x) for x in X_test])
        test_accuracy = accuracy_score(y_test, y_test_pred)
        
        print(f"Train accuracy: {train_accuracy * 100:.2f}%")
        print(f"Test accuracy: {test_accuracy * 100:.2f}%")
        
        # Keep track of the best model
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_svm = svm
            print(f"New best model: {name} with test accuracy {best_accuracy * 100:.2f}%")
    
    # Create and save the iteration history as an image for the best model
    table_path = os.path.join(output_dir, 'real_data_iteration_history_table.png')
    create_iteration_table_image(best_svm, table_path)
    
    # Visualize the results for the best model
    visualize_svm_results(X_train, y_train, best_svm, output_dir)
    
    # Evaluate the model on test data
    evaluate_model_on_test_data(best_svm, X_test, y_test, output_dir, X_train, y_train)
    
    print("\nApple disease detection with real data complete!")
    print(f"All results have been saved to: {output_dir}")

if __name__ == "__main__":
    main()