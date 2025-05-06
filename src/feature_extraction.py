import os
import numpy as np
import cv2
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

# Define paths
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATASET_DIR = os.path.join(PROJECT_DIR, 'data', 'dataset', 'apple_disease', 'apple')
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'data', 'features')

# Make sure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_image(image_path):
    """
    Load an image from the specified path
    
    Parameters:
    -----------
    image_path : str
        Path to the image file
    
    Returns:
    --------
    numpy.ndarray
        Loaded image
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image at {image_path}")
    return img

def preprocess_image(img, target_size=(224, 224)):
    """
    Preprocess the image for feature extraction
    
    Parameters:
    -----------
    img : numpy.ndarray
        Input image
    target_size : tuple
        Target size for resizing the image
    
    Returns:
    --------
    numpy.ndarray
        Preprocessed image
    """
    # Resize image
    img_resized = cv2.resize(img, target_size)
    
    # Convert to RGB (OpenCV loads as BGR)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    # Normalize pixel values to range [0, 1]
    img_normalized = img_rgb / 255.0
    
    return img_normalized

def extract_color_features(img):
    """
    Extract color-based features from an image
    
    Parameters:
    -----------
    img : numpy.ndarray
        Input RGB image (normalized)
    
    Returns:
    --------
    numpy.ndarray
        Color features vector
    """
    # Calculate mean and standard deviation for each channel
    r_mean, r_std = np.mean(img[:,:,0]), np.std(img[:,:,0])
    g_mean, g_std = np.mean(img[:,:,1]), np.std(img[:,:,1])
    b_mean, b_std = np.mean(img[:,:,2]), np.std(img[:,:,2])
    
    # Calculate color histogram
    hist_r = cv2.calcHist([img], [0], None, [8], [0, 1])
    hist_g = cv2.calcHist([img], [1], None, [8], [0, 1])
    hist_b = cv2.calcHist([img], [2], None, [8], [0, 1])
    
    # Normalize histograms
    hist_r = cv2.normalize(hist_r, hist_r).flatten()
    hist_g = cv2.normalize(hist_g, hist_g).flatten()
    hist_b = cv2.normalize(hist_b, hist_b).flatten()
    
    # Combine features
    color_features = np.concatenate([
        [r_mean, r_std, g_mean, g_std, b_mean, b_std],
        hist_r, hist_g, hist_b
    ])
    
    return color_features

def extract_texture_features(img):
    """
    Extract texture-based features from an image using GLCM
    
    Parameters:
    -----------
    img : numpy.ndarray
        Input RGB image (normalized)
    
    Returns:
    --------
    numpy.ndarray
        Texture features vector
    """
    # Convert to grayscale
    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    
    # Calculate Haralick textures
    haralick_features = []
    
    # We'll use a simple approach by calculating GLCM at 4 angles
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    distances = [1, 3]
    
    for angle in angles:
        for distance in distances:
            dx = int(np.cos(angle) * distance)
            dy = int(np.sin(angle) * distance)
            
            # Create the GLCM matrix
            glcm = np.zeros((8, 8), dtype=np.uint32)
            quantized = gray // 32  # Quantize to 8 levels
            
            height, width = quantized.shape
            for i in range(height):
                for j in range(width):
                    ni, nj = i + dy, j + dx
                    if 0 <= ni < height and 0 <= nj < width:
                        glcm[quantized[i, j], quantized[ni, nj]] += 1
            
            # Normalize GLCM
            if glcm.sum() > 0:
                glcm = glcm / glcm.sum()
                
            # Calculate features from GLCM
            # Contrast
            contrast = np.sum(np.square(np.arange(8) - np.arange(8).reshape(-1, 1)) * glcm)
            
            # Homogeneity
            homogeneity = np.sum(glcm / (1 + np.square(np.arange(8) - np.arange(8).reshape(-1, 1))))
            
            # Energy
            energy = np.sum(np.square(glcm))
            
            # Correlation
            mu_i = np.sum(np.arange(8).reshape(-1, 1) * glcm)
            mu_j = np.sum(np.arange(8) * glcm)
            sigma_i = np.sqrt(np.sum(np.square(np.arange(8).reshape(-1, 1) - mu_i) * glcm))
            sigma_j = np.sqrt(np.sum(np.square(np.arange(8) - mu_j) * glcm))
            
            correlation = 0
            if sigma_i > 0 and sigma_j > 0:
                correlation = np.sum((np.arange(8).reshape(-1, 1) - mu_i) * (np.arange(8) - mu_j) * glcm) / (sigma_i * sigma_j)
            
            haralick_features.extend([contrast, homogeneity, energy, correlation])
    
    # Extract some edge features using Sobel filters
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Calculate magnitude and direction
    magnitude = np.sqrt(np.square(sobel_x) + np.square(sobel_y))
    
    # Calculate statistics of the gradient
    edge_mean = np.mean(magnitude)
    edge_std = np.std(magnitude)
    edge_max = np.max(magnitude)
    
    # Combine all texture features
    texture_features = np.array(haralick_features + [edge_mean, edge_std, edge_max])
    
    return texture_features

def extract_shape_features(img):
    """
    Extract shape-based features from an image
    
    Parameters:
    -----------
    img : numpy.ndarray
        Input RGB image (normalized)
    
    Returns:
    --------
    numpy.ndarray
        Shape features vector
    """
    # Convert to grayscale
    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    
    # Threshold the image to get binary mask (this is simplified and would need tuning)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Initialize shape features with zeros (in case no contours are found)
    shape_features = np.zeros(5)
    
    if contours:
        # Get the largest contour (assuming it's the main subject)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Calculate shape features
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        
        # Avoid division by zero
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter * perimeter)
        else:
            circularity = 0
            
        # Fit an ellipse to the contour
        if len(largest_contour) >= 5:
            ellipse = cv2.fitEllipse(largest_contour)
            (_, _), (ma, mi), _ = ellipse
            
            # Calculate aspect ratio
            if mi > 0:
                aspect_ratio = ma / mi
            else:
                aspect_ratio = 1
                
            # Eccentricity
            ecc = np.sqrt(1 - (mi/ma)**2) if ma > 0 else 0
            
        else:
            aspect_ratio = 1
            ecc = 0
            
        # Extent = contour area / bounding rect area
        x, y, w, h = cv2.boundingRect(largest_contour)
        if w * h > 0:
            extent = area / (w * h)
        else:
            extent = 0
            
        shape_features = np.array([area, circularity, aspect_ratio, ecc, extent])
    
    return shape_features

def extract_all_features(img):
    """
    Extract all features from an image
    
    Parameters:
    -----------
    img : numpy.ndarray
        Input image
    
    Returns:
    --------
    numpy.ndarray
        Combined feature vector
    """
    # Preprocess the image
    processed_img = preprocess_image(img)
    
    # Extract individual feature sets
    color_features = extract_color_features(processed_img)
    texture_features = extract_texture_features(processed_img)
    shape_features = extract_shape_features(processed_img)
    
    # Combine all features
    all_features = np.concatenate([color_features, texture_features, shape_features])
    
    return all_features

def extract_features_from_dataset(num_samples=None):
    """
    Extract features from all images in the dataset
    
    Parameters:
    -----------
    num_samples : int, optional
        Number of samples to process per class (None = all)
    
    Returns:
    --------
    tuple
        X (features), y (labels), and file_paths
    """
    features_list = []
    labels = []
    file_paths = []
    
    # Define the classes: apple scab (disease) vs healthy
    classes = {
        'apple scab': -1,  # Disease label = -1 (following the convention in apple_disease_detection.py)
        'healthy': 1     # Healthy label = 1
    }
    
    # Process each class
    for class_name, label in classes.items():
        class_dir = os.path.join(DATASET_DIR, class_name)
        if not os.path.exists(class_dir):
            print(f"Warning: Directory not found: {class_dir}")
            continue
            
        image_files = [f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.jpeg', '.png', '.JPG'))]
        
        if num_samples is not None:
            image_files = image_files[:num_samples]
            
        print(f"Processing {len(image_files)} images from class '{class_name}'")
        
        for img_file in image_files:
            img_path = os.path.join(class_dir, img_file)
            
            try:
                # Load and process image
                img = load_image(img_path)
                features = extract_all_features(img)
                
                # Store results
                features_list.append(features)
                labels.append(label)
                file_paths.append(img_path)
                
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
    
    # Convert to numpy arrays
    X = np.array(features_list)
    y = np.array(labels)
    
    # Normalize features using StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, file_paths

def visualize_feature_distribution(X, y, feature_names=None):
    """
    Visualize the distribution of features between classes
    
    Parameters:
    -----------
    X : numpy.ndarray
        Feature matrix
    y : numpy.ndarray
        Labels
    feature_names : list, optional
        Names of features
    
    Returns:
    --------
    str
        Path to saved visualization
    """
    # Select a subset of features to visualize (the first 6 for example)
    num_features = min(6, X.shape[1])
    
    if feature_names is None:
        feature_names = [f"Feature {i+1}" for i in range(num_features)]
    else:
        feature_names = feature_names[:num_features]
    
    plt.figure(figsize=(15, 10))
    
    for i in range(num_features):
        plt.subplot(2, 3, i+1)
        
        # Split features by class
        healthy_values = X[y == 1, i]
        diseased_values = X[y == -1, i]
        
        # Plot histograms
        plt.hist(healthy_values, alpha=0.5, bins=20, label='Healthy', color='green')
        plt.hist(diseased_values, alpha=0.5, bins=20, label='Apple Scab', color='red')
        
        plt.title(feature_names[i])
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.legend()
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(OUTPUT_DIR, 'feature_distribution.png')
    plt.savefig(output_path)
    plt.close()
    
    return output_path

def visualize_sample_processing(image_path):
    """
    Visualize the feature extraction process on a sample image
    
    Parameters:
    -----------
    image_path : str
        Path to the sample image
    
    Returns:
    --------
    str
        Path to saved visualization
    """
    # Load image
    img = load_image(image_path)
    
    # Create figure
    plt.figure(figsize=(20, 10))
    
    # Original image
    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis('off')
    
    # Preprocessed image
    processed = preprocess_image(img)
    plt.subplot(2, 3, 2)
    plt.imshow(processed)
    plt.title("Preprocessed Image")
    plt.axis('off')
    
    # Grayscale image
    gray = cv2.cvtColor((processed * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    plt.subplot(2, 3, 3)
    plt.imshow(gray, cmap='gray')
    plt.title("Grayscale Image")
    plt.axis('off')
    
    # Edge detection (for texture features)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(np.square(sobel_x) + np.square(sobel_y))
    
    plt.subplot(2, 3, 4)
    plt.imshow(magnitude, cmap='viridis')
    plt.title("Edge Detection")
    plt.axis('off')
    
    # Binary image (for shape features)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    plt.subplot(2, 3, 5)
    plt.imshow(binary, cmap='gray')
    plt.title("Binary Image")
    plt.axis('off')
    
    # Color histogram
    plt.subplot(2, 3, 6)
    for i, color in enumerate(['r', 'g', 'b']):
        hist = cv2.calcHist([processed], [i], None, [256], [0, 1])
        plt.plot(hist, color=color)
    plt.title("Color Histogram")
    plt.xlim([0, 1])
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(OUTPUT_DIR, 'sample_processing.png')
    plt.savefig(output_path)
    plt.close()
    
    # Extract features
    color_features = extract_color_features(processed)
    texture_features = extract_texture_features(processed)
    shape_features = extract_shape_features(processed)
    
    # Create a feature summary
    features_df = pd.DataFrame({
        'Feature Type': ['Color Mean R', 'Color Std R', 'Color Mean G', 'Color Std G', 
                         'Color Mean B', 'Color Std B', 'Texture Contrast', 'Texture Homogeneity',
                         'Texture Energy', 'Edge Mean', 'Edge Std', 'Shape Area', 'Shape Circularity'],
        'Value': [color_features[0], color_features[1], color_features[2], color_features[3],
                 color_features[4], color_features[5], texture_features[0], texture_features[1],
                 texture_features[2], texture_features[-3], texture_features[-2], 
                 shape_features[0], shape_features[1]]
    })
    
    # Save feature summary
    summary_path = os.path.join(OUTPUT_DIR, 'feature_summary.csv')
    features_df.to_csv(summary_path, index=False)
    
    return output_path, summary_path

def save_features_to_csv(X, y, file_paths):
    """
    Save extracted features and labels to CSV file
    
    Parameters:
    -----------
    X : numpy.ndarray
        Feature matrix
    y : numpy.ndarray
        Labels
    file_paths : list
        List of file paths for each sample
    
    Returns:
    --------
    str
        Path to saved CSV file
    """
    # Create a DataFrame
    df = pd.DataFrame(X)
    
    # Add labels and file paths
    df['label'] = y
    df['file_path'] = file_paths
    
    # Add class name based on label
    df['class'] = df['label'].apply(lambda x: 'Healthy' if x == 1 else 'Apple Scab')
    
    # Save to CSV
    output_path = os.path.join(OUTPUT_DIR, 'apple_disease_features.csv')
    df.to_csv(output_path, index=False)
    
    return output_path

def main():
    """
    Main function to demonstrate feature extraction
    """
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Starting feature extraction process...")
    
    # Extract features from a limited number of samples from each class
    X, y, file_paths = extract_features_from_dataset(num_samples=20)
    
    print(f"\nExtracted features from {len(file_paths)} images")
    print(f"Feature vector shape: {X.shape}")
    
    # Save features to CSV
    csv_path = save_features_to_csv(X, y, file_paths)
    print(f"\nFeatures saved to: {csv_path}")
    
    # Visualize feature distribution
    dist_path = visualize_feature_distribution(X, y)
    print(f"\nFeature distribution visualization saved to: {dist_path}")
    
    # Process and visualize a sample image (first apple scab image)
    apple_scab_samples = [fp for fp, label in zip(file_paths, y) if label == -1]
    if apple_scab_samples:
        sample_path = apple_scab_samples[0]
        viz_path, summary_path = visualize_sample_processing(sample_path)
        print(f"\nSample processing visualization saved to: {viz_path}")
        print(f"Sample feature summary saved to: {summary_path}")
    
    print("\nFeature extraction complete!")

if __name__ == "__main__":
    main()