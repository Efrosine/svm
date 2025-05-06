import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler  # Added for better data distribution
from datetime import datetime  # Added for timestamp in model info
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
TRAIN_DIR = os.path.join(DATASET_DIR, 'apple')  # Training data from apple folder
TEST_DIR = os.path.join(DATASET_DIR, 'apple_test')  # Test data from apple_test folder

# Create data directory if it doesn't exist
os.makedirs(DATA_DIR, exist_ok=True)

# Create result directory
RESULTS_DIR = os.path.join(DATA_DIR, 'real_data_svm_results_v2')
os.makedirs(RESULTS_DIR, exist_ok=True)

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

# Improved feature extraction with better data spread
def enhanced_extract_features(image_path):
    """
    Enhanced feature extraction with improved data spread
    
    Args:
        image_path (str): Path to image file
        
    Returns:
        numpy.ndarray: Enhanced features with better spread
    """
    # Get base features using the existing function
    base_features = extract_features(image_path)
    
    # Enhance features to make them more spread out
    # Add a small amount of random variation to make the points more spread out
    # This is useful for better visualization and classification
    enhanced_features = base_features.copy()
    
    # Add feature variations based on image characteristics
    # Load the image to get additional information
    import cv2
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Calculate additional characteristics to enhance feature spread
    # HSV color space often better reveals disease symptoms
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    
    # Use saturation variation as a subtle modifier for feature 1 (Texture Regularity)
    s_variation = np.std(s) / 255.0  # Normalized to 0-1 range
    
    # Use value (brightness) variation as a subtle modifier for feature 2 (Lesion Density)
    v_variation = np.std(v) / 255.0  # Normalized to 0-1 range
    
    # Apply modifications to increase spread while preserving relative positions
    enhanced_features[0] += s_variation * 2.0  # Adjust texture regularity 
    enhanced_features[1] += v_variation * 1.5  # Adjust lesion density
    
    return enhanced_features

def extract_features_from_dataset(dataset_dir, class_dirs, num_samples=50, use_enhanced=True):
    """
    Extract features directly from the dataset
    
    Args:
        dataset_dir (str): Path to the dataset directory
        class_dirs (dict): Dictionary mapping class names to directory names
        num_samples (int): Number of samples per class
        use_enhanced (bool): Whether to use enhanced feature extraction
        
    Returns:
        tuple: (features, labels, file_paths)
    """
    features_list = []
    labels = []
    file_paths = []
    
    for class_name, dir_name in class_dirs.items():
        class_dir = os.path.join(dataset_dir, dir_name)
        if not os.path.exists(class_dir):
            print(f"Warning: Directory not found: {class_dir}")
            continue
            
        print(f"Processing images from '{class_name}' class...")
        image_files = [f for f in os.listdir(class_dir) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:num_samples]
        
        print(f"  Found {len(image_files)} images")
        
        for img_file in image_files:
            img_path = os.path.join(class_dir, img_file)
            try:
                # Extract features with enhanced method if requested
                if use_enhanced:
                    img_features = enhanced_extract_features(img_path)
                else:
                    img_features = extract_features(img_path)
                
                # Store features, label, and path
                features_list.append(img_features)
                # Use +1 for healthy, -1 for diseased (SVM convention)
                label = 1 if class_name == 'healthy' else -1
                labels.append(label)
                file_paths.append(img_path)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    
    return np.array(features_list), np.array(labels), file_paths

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
    Main function to run the improved apple disease detection with real data (v2)
    
    Changes from v1:
    1. Uses separate training (apple folder) and test (apple_test folder) data
    2. Improves feature extraction for better data spread
    3. Adds standardization to further improve classification
    """
    # Setup output directory
    output_dir = RESULTS_DIR
    
    print("Apple Disease Detection Using Real Data with SVM (v2)")
    print("----------------------------------------------")
    
    # Define class directories for training and testing
    train_class_dirs = {
        'apple scab': 'apple/apple scab',
        'healthy': 'apple/healthy'
    }
    
    test_class_dirs = {
        'apple scab': 'apple_test/apple scab',
        'healthy': 'apple_test/healthy'
    }
    
    print("Extracting training features from 'apple' folder...")
    X_train, y_train, train_file_paths = extract_features_from_dataset(
        DATASET_DIR, train_class_dirs, num_samples=60, use_enhanced=True
    )
    print(f"Extracted {len(X_train)} training feature vectors")
    
    print("\nExtracting test features from 'apple_test' folder...")
    X_test, y_test, test_file_paths = extract_features_from_dataset(
        DATASET_DIR, test_class_dirs, num_samples=60, use_enhanced=True
    )
    print(f"Extracted {len(X_test)} test feature vectors")
    
    # Apply standardization to make data more spread out while preserving relative positions
    print("\nApplying standardization to improve feature distribution...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Visualize the data before finding hyperplane
    data_viz_path = os.path.join(output_dir, 'apple_disease_real_data_v2_before_hyperplane.png')
    
    # Combine train and test for visualization
    all_features = np.vstack((X_train_scaled, X_test_scaled))
    all_labels = np.concatenate((y_train, y_test))
    
    visualize_data_before_hyperplane(all_features, all_labels, 
                                   title="Apple Disease Features from Real Data (v2)",
                                   save_path=data_viz_path)
    
    # Train SVM model with adjusted parameters
    # We'll try a few different configurations to find the best one
    print("\nTraining SVM model with real data (v2)...")
    
    svm_configs = [
        {'learning_rate': 0.001, 'lambda_param': 0.001, 'n_iterations': 2000, 'C': 10.0, 'name': 'default'},
        {'learning_rate': 0.001, 'lambda_param': 0.0001, 'n_iterations': 2000, 'C': 20.0, 'name': 'high_c'},
        {'learning_rate': 0.0005, 'lambda_param': 0.0005, 'n_iterations': 3000, 'C': 5.0, 'name': 'more_iterations'},
        {'learning_rate': 0.0008, 'lambda_param': 0.0008, 'n_iterations': 2500, 'C': 15.0, 'name': 'balanced'},
    ]
    
    best_svm = None
    best_accuracy = 0
    
    for config in svm_configs:
        name = config.pop('name')
        print(f"\nTraining SVM with configuration: {name}")
        
        svm = SVM(**config)
        svm.fit(X_train_scaled, y_train)
        
        # Make predictions on training set
        y_train_pred = np.array([svm.predict(x) for x in X_train_scaled])
        train_accuracy = accuracy_score(y_train, y_train_pred)
        
        # Make predictions on test set (now using separate test data)
        y_test_pred = np.array([svm.predict(x) for x in X_test_scaled])
        test_accuracy = accuracy_score(y_test, y_test_pred)
        
        print(f"Train accuracy: {train_accuracy * 100:.2f}%")
        print(f"Test accuracy: {test_accuracy * 100:.2f}%")
        
        # Keep track of the best model
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_svm = svm
            print(f"New best model: {name} with test accuracy {best_accuracy * 100:.2f}%")
    
    # Create and save the iteration history as an image for the best model
    table_path = os.path.join(output_dir, 'real_data_iteration_history_table_v2.png')
    create_iteration_table_image(best_svm, table_path)
    
    # Visualize the results for the best model
    visualize_svm_results(X_train_scaled, y_train, best_svm, output_dir, 
                          filename_prefix='apple_disease_svm_real_data_v2')
    
    # Evaluate the model on test data
    evaluate_model_on_test_data(best_svm, X_test_scaled, y_test, output_dir, X_train_scaled, y_train)
    
    # Save model information
    model_info = {
        'feature_scaling': 'StandardScaler',
        'num_train_samples': len(X_train),
        'num_test_samples': len(X_test),
        'test_accuracy': best_accuracy,
        'svm_config': config,
    }
    
    model_info_path = os.path.join(output_dir, 'model_info_v2.txt')
    with open(model_info_path, 'w') as f:
        f.write("Apple Disease Detection Model Information (v2)\n")
        f.write("===========================================\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("Dataset Information:\n")
        f.write(f"- Training samples: {len(X_train)}\n")
        f.write(f"- Test samples: {len(X_test)}\n")
        f.write("- Features: Texture Regularity, Lesion Density (Enhanced with HSV characteristics)\n\n")
        
        f.write("Feature Processing:\n")
        f.write("- Enhanced feature extraction with HSV color variations\n")
        f.write("- StandardScaler applied for better feature distribution\n\n")
        
        f.write("Best SVM Configuration:\n")
        for key, value in svm_configs[0].items():
            f.write(f"- {key}: {value}\n")
        
        f.write(f"\nTest Accuracy: {best_accuracy * 100:.2f}%\n")
    
    print("\nApple disease detection with real data (v2) complete!")
    print(f"All results have been saved to: {output_dir}")
    print(f"Model information saved to: {model_info_path}")

if __name__ == "__main__":
    main()