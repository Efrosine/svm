import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

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

# Create results directory
RESULTS_DIR = os.path.join(DATA_DIR, 'real_data_svm_results_v2')

def extract_features_from_dataset(dataset_dir, class_dirs, num_samples=80, use_enhanced=True):
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
    # Reusing the function from apple_disease_detection_real_data_v2.py
    from apple_disease_detection_real_data_v2 import enhanced_extract_features
    
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

def find_correctly_classified_samples():
    """
    Find 10 correctly classified samples from each class that are in the correct area
    (not crossing the hyperplane) and save them to a CSV file.
    """
    # Define class directories for training and testing
    train_class_dirs = {
        'apple scab': 'apple/apple scab',
        'healthy': 'apple/healthy'
    }
    
    test_class_dirs = {
        'apple scab': 'apple_test/apple scab',
        'healthy': 'apple_test/healthy'
    }
    
    # Extract features from the training dataset (increased sample count to 80)
    print("Extracting features from training dataset...")
    X_train, y_train, train_file_paths = extract_features_from_dataset(
        DATASET_DIR, train_class_dirs, num_samples=80, use_enhanced=True
    )
    
    # Also extract test features to have more samples to choose from
    print("\nExtracting features from test dataset...")
    X_test, y_test, test_file_paths = extract_features_from_dataset(
        DATASET_DIR, test_class_dirs, num_samples=80, use_enhanced=True
    )
    
    # Combine training and test data to have more samples to select from
    X_combined = np.vstack((X_train, X_test))
    y_combined = np.concatenate((y_train, y_test))
    file_paths_combined = train_file_paths + test_file_paths
    
    print(f"Combined dataset: {len(X_combined)} samples")
    
    # Apply standardization to all data
    print("Applying standardization...")
    scaler = StandardScaler()
    X_combined_scaled = scaler.fit_transform(X_combined)
    
    # Train SVM model with adjusted parameters (might need to adjust C or lambda)
    # Try different configurations to ensure we get enough samples in each class
    svm_configs = [
        {'learning_rate': 0.001, 'lambda_param': 0.001, 'n_iterations': 2000, 'C': 10.0},
        {'learning_rate': 0.001, 'lambda_param': 0.0005, 'n_iterations': 2000, 'C': 5.0},
        {'learning_rate': 0.001, 'lambda_param': 0.001, 'n_iterations': 2000, 'C': 15.0}
    ]
    
    best_config = None
    most_samples = 0
    
    for i, config in enumerate(svm_configs):
        print(f"\nTraining SVM model with configuration {i+1}...")
        svm_model = SVM(**config)
        svm_model.fit(X_combined_scaled, y_combined)
        
        # Calculate functional margins for all samples
        functional_margins = np.array([
            y_i * (np.dot(svm_model.w, x_i) - svm_model.b) 
            for x_i, y_i in zip(X_combined_scaled, y_combined)
        ])
        
        # Use different thresholds for each class to find enough samples
        # Start with the ideal threshold (> 1) and gradually relax if needed
        healthy_threshold = 1.0  # Standard threshold
        diseased_threshold = 0.8  # Slightly relaxed to find more diseased samples
        
        # Find samples that are correctly classified with good margins
        healthy_indices = np.where((y_combined == 1) & (functional_margins > healthy_threshold))[0]
        diseased_indices = np.where((y_combined == -1) & (functional_margins > diseased_threshold))[0]
        
        print(f"  Config {i+1}: Found {len(healthy_indices)} healthy and {len(diseased_indices)} diseased samples")
        
        # Keep track of the configuration that gives the most samples from the minority class
        min_samples = min(len(healthy_indices), len(diseased_indices))
        if min_samples > most_samples:
            most_samples = min_samples
            best_config = i
    
    # Use the best configuration
    print(f"\nUsing best configuration (config {best_config+1}) for final selection...")
    svm_model = SVM(**svm_configs[best_config])
    svm_model.fit(X_combined_scaled, y_combined)
    
    # Calculate functional margins with the best model
    functional_margins = np.array([
        y_i * (np.dot(svm_model.w, x_i) - svm_model.b) 
        for x_i, y_i in zip(X_combined_scaled, y_combined)
    ])
    
    # Start with the standard threshold (functional margin > 1)
    threshold = 1.0
    
    # Find samples with good margins
    healthy_indices = np.where((y_combined == 1) & (functional_margins > threshold))[0]
    diseased_indices = np.where((y_combined == -1) & (functional_margins > threshold))[0]
    
    # If we don't have enough diseased samples, gradually relax the threshold
    if len(diseased_indices) < 10:
        for relaxed_threshold in [0.9, 0.8, 0.7, 0.6, 0.5]:
            diseased_indices = np.where((y_combined == -1) & (functional_margins > relaxed_threshold))[0]
            if len(diseased_indices) >= 10:
                print(f"Relaxed threshold for diseased samples to {relaxed_threshold} to find enough samples")
                break
    
    # Sort indices by margin (highest margin first)
    healthy_indices = healthy_indices[np.argsort(functional_margins[healthy_indices])[::-1]]
    diseased_indices = diseased_indices[np.argsort(functional_margins[diseased_indices])[::-1]]
    
    # Select 10 samples from each class (or all if less than 10)
    selected_healthy = healthy_indices[:min(10, len(healthy_indices))]
    selected_diseased = diseased_indices[:min(10, len(diseased_indices))]
    
    # Combine the selected samples
    selected_indices = np.concatenate([selected_healthy, selected_diseased])
    
    # Create a DataFrame with the selected samples
    sample_data = []
    for idx in selected_indices:
        file_path = file_paths_combined[idx]
        class_name = 'healthy' if y_combined[idx] == 1 else 'apple scab'
        functional_margin = functional_margins[idx]
        
        # Extract just the filename from the path
        filename = os.path.basename(file_path)
        
        # Determine if sample is from training or test set
        dataset_source = "Train" if idx < len(X_train) else "Test"
        
        sample_data.append({
            'Filename': filename,
            'Class': class_name,
            'Functional_Margin': functional_margin,
            'Source': dataset_source,
            'File_Path': file_path,
            'Feature_1': X_combined[idx, 0],  # Original (unscaled) feature 1
            'Feature_2': X_combined[idx, 1],  # Original (unscaled) feature 2
            'Scaled_Feature_1': X_combined_scaled[idx, 0],  # Scaled feature 1
            'Scaled_Feature_2': X_combined_scaled[idx, 1]   # Scaled feature 2
        })
    
    # Create DataFrame
    df = pd.DataFrame(sample_data)
    
    # Save to CSV
    csv_path = os.path.join(RESULTS_DIR, 'correctly_classified_samples.csv')
    df.to_csv(csv_path, index=False)
    
    print(f"\nSelected {len(selected_healthy)} healthy samples and {len(selected_diseased)} diseased samples.")
    print(f"Saved {len(df)} correctly classified samples to {csv_path}")
    
    # Create a visualization of the selected samples
    plt.figure(figsize=(12, 10))
    
    # Plot all data as small points
    healthy_all = (y_combined == 1)
    diseased_all = (y_combined == -1)
    
    plt.scatter(X_combined_scaled[healthy_all, 0], X_combined_scaled[healthy_all, 1], 
                color='lightgreen', marker='o', label='All Healthy', 
                s=50, alpha=0.3)
    plt.scatter(X_combined_scaled[diseased_all, 0], X_combined_scaled[diseased_all, 1], 
                color='lightcoral', marker='x', label='All Diseased', 
                s=50, alpha=0.3)
    
    # Plot selected samples as larger points
    for idx in selected_healthy:
        plt.scatter(X_combined_scaled[idx, 0], X_combined_scaled[idx, 1], 
                    color='darkgreen', marker='o', s=200, alpha=1.0, 
                    edgecolors='black', linewidths=2)
    
    for idx in selected_diseased:
        plt.scatter(X_combined_scaled[idx, 0], X_combined_scaled[idx, 1], 
                    color='darkred', marker='x', s=200, alpha=1.0, 
                    linewidths=3)
    
    # Add a legend for selected samples
    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], color='darkgreen', marker='o', markersize=10, 
               markeredgecolor='black', linestyle='None', label='Selected Healthy (10)'),
        Line2D([0], [0], color='darkred', marker='x', markersize=10, 
               linestyle='None', linewidth=2, label='Selected Diseased (10)')
    ]
    plt.legend(handles=custom_lines, loc='upper right', fontsize=12)
    
    # Plot decision boundary
    x_min, x_max = X_combined_scaled[:, 0].min() - 0.5, X_combined_scaled[:, 0].max() + 0.5
    y_min, y_max = X_combined_scaled[:, 1].min() - 0.5, X_combined_scaled[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    Z = np.array([svm_model.predict(np.array([x, y])) for x, y in zip(xx.ravel(), yy.ravel())])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.2, cmap=plt.cm.RdYlGn)
    
    # Plot decision boundary line and margins
    w_norm = np.linalg.norm(svm_model.w)
    if w_norm > 0:
        xx_margin = np.linspace(x_min, x_max, 100)
        yy_db = (-svm_model.w[0] * xx_margin + svm_model.b) / svm_model.w[1]
        plt.plot(xx_margin, yy_db, 'k-', linewidth=2, label='Decision Boundary')
        
        # Add the margin lines
        yy_pos_margin = (-svm_model.w[0] * xx_margin + svm_model.b + 1) / svm_model.w[1]
        plt.plot(xx_margin, yy_pos_margin, 'k--', linewidth=1.5)
        
        yy_neg_margin = (-svm_model.w[0] * xx_margin + svm_model.b - 1) / svm_model.w[1]
        plt.plot(xx_margin, yy_neg_margin, 'k--', linewidth=1.5, label='Margin')
    
    plt.xlabel('Texture Regularity (Scaled)', fontsize=14)
    plt.ylabel('Lesion Density (Scaled)', fontsize=14)
    plt.title('Selected 10 Samples from Each Class (Beyond Margin)', fontsize=16)
    plt.grid(True, alpha=0.3)
    
    # Save the visualization
    viz_path = os.path.join(RESULTS_DIR, 'correctly_classified_samples_visualization.png')
    plt.savefig(viz_path, dpi=300)
    plt.close()
    
    print(f"Visualization saved to {viz_path}")

if __name__ == "__main__":
    find_correctly_classified_samples()