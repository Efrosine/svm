import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import pandas as pd

def load_image(image_path, target_size=(128, 128)):
    """
    Load and preprocess an image for feature extraction
    
    Args:
        image_path (str): Path to the image file
        target_size (tuple): Target size for resizing
        
    Returns:
        numpy.ndarray: Preprocessed image
    """
    # Load the image
    img = cv2.imread(image_path)
    
    # Convert from BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize image to target size
    img = cv2.resize(img, target_size)
    
    return img

def extract_color_features(image):
    """
    Extract color features from an image (mean and std of each channel)
    
    Args:
        image (numpy.ndarray): Input image
        
    Returns:
        numpy.ndarray: Color features
    """
    # Calculate mean and std for each channel
    means = np.mean(image, axis=(0, 1))
    stds = np.std(image, axis=(0, 1))
    
    # Concatenate features
    color_features = np.concatenate([means, stds])
    
    return color_features

def extract_texture_features(image):
    """
    Extract texture features using Haralick texture features
    
    Args:
        image (numpy.ndarray): Input image
        
    Returns:
        numpy.ndarray: Texture features
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Calculate GLCM (Gray-Level Co-occurrence Matrix)
    glcm = np.zeros((8, 8), dtype=np.float32)
    h, w = gray.shape
    for i in range(h-1):
        for j in range(w-1):
            i_idx = min(gray[i, j] // 32, 7)       # Quantize to 8 levels
            j_idx = min(gray[i+1, j+1] // 32, 7)   # Diagonal neighbor
            glcm[i_idx, j_idx] += 1
    
    # Normalize GLCM
    if glcm.sum() > 0:
        glcm /= glcm.sum()
    
    # Extract simple statistics from GLCM
    contrast = np.sum(np.square(np.arange(8) - np.arange(8)[:, np.newaxis]) * glcm)
    energy = np.sum(np.square(glcm))
    homogeneity = np.sum(glcm / (1 + np.square(np.arange(8) - np.arange(8)[:, np.newaxis])))
    
    # Calculate additional texture features
    # Entropy
    epsilon = 1e-10  # Small constant to avoid log(0)
    entropy = -np.sum(glcm * np.log(glcm + epsilon))
    
    return np.array([contrast, energy, homogeneity, entropy])

def extract_shape_features(image):
    """
    Extract shape features using contours and edge detection
    
    Args:
        image (numpy.ndarray): Input image
        
    Returns:
        numpy.ndarray: Shape features
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Apply Canny edge detection
    edges = cv2.Canny(gray, 100, 200)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Calculate shape features
    num_contours = len(contours)
    
    # If no contours found, return zeros
    if num_contours == 0:
        return np.zeros(4)
    
    # Get the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    contour_area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, True)
    
    # Calculate circularity and aspect ratio
    circularity = 0
    if perimeter > 0:
        circularity = 4 * np.pi * contour_area / (perimeter * perimeter)
    
    # Bounding rectangle
    x, y, w, h = cv2.boundingRect(largest_contour)
    aspect_ratio = w / h if h > 0 else 0
    
    return np.array([num_contours, contour_area, circularity, aspect_ratio])

def extract_features(image_path, visualize=False):
    """
    Extract features from an image
    
    Args:
        image_path (str): Path to the image file
        visualize (bool): Whether to visualize the process
        
    Returns:
        tuple: (features, visualization if requested)
    """
    # Load and preprocess the image
    img = load_image(image_path)
    
    # Extract features
    color_features = extract_color_features(img)
    texture_features = extract_texture_features(img)
    shape_features = extract_shape_features(img)
    
    # Combine all features
    features = np.concatenate([color_features, texture_features, shape_features])
    
    # Create visualization if requested
    if visualize:
        # Original image
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        plt.imshow(img)
        plt.title('Original Image')
        
        # Grayscale for texture
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        plt.subplot(2, 2, 2)
        plt.imshow(gray, cmap='gray')
        plt.title('Grayscale (for texture)')
        
        # Edges for shape features
        edges = cv2.Canny(gray, 100, 200)
        plt.subplot(2, 2, 3)
        plt.imshow(edges, cmap='gray')
        plt.title('Edge Detection (for shape)')
        
        # Color channels
        plt.subplot(2, 2, 4)
        r_channel = img[:,:,0]
        g_channel = img[:,:,1]
        b_channel = img[:,:,2]
        plt.bar(['R-mean', 'G-mean', 'B-mean', 'R-std', 'G-std', 'B-std'], 
                color_features, color=['red', 'green', 'blue', 'darkred', 'darkgreen', 'darkblue'])
        plt.title('Color Features')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        return features, plt.gcf()
    
    return features

def load_dataset(dataset_path, class_name="apple scab", max_samples=None):
    """
    Load the dataset and extract features
    
    Args:
        dataset_path (str): Path to the dataset directory
        class_name (str): Name of the class to focus on (disease class)
        max_samples (int): Maximum number of samples to load (for testing)
        
    Returns:
        tuple: (features, labels, file_paths)
    """
    features_list = []
    labels = []
    file_paths = []
    
    apple_dir = os.path.join(dataset_path, "apple")
    
    # Find all subdirectories (classes)
    class_dirs = sorted([d for d in os.listdir(apple_dir) 
                  if os.path.isdir(os.path.join(apple_dir, d))])
    
    # Process each class
    for class_idx, class_dir in enumerate(class_dirs):
        class_path = os.path.join(apple_dir, class_dir)
        
        # 1 for the target disease class, 0 for others
        label = 1 if class_dir.lower() == class_name.lower() else 0
        
        # Get image files in the class directory
        image_files = [f for f in os.listdir(class_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Limit number of samples if specified
        if max_samples:
            image_files = image_files[:max_samples]
        
        print(f"Processing {len(image_files)} images from class '{class_dir}'...")
        
        # Process each image
        for img_file in image_files:
            img_path = os.path.join(class_path, img_file)
            try:
                # Extract features
                img_features = extract_features(img_path)
                
                # Store features and label
                features_list.append(img_features)
                labels.append(label)
                file_paths.append(img_path)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    
    return np.array(features_list), np.array(labels), file_paths

def normalize_features(features):
    """
    Normalize features using StandardScaler
    
    Args:
        features (numpy.ndarray): Features array
        
    Returns:
        numpy.ndarray: Normalized features
    """
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(features)
    return normalized_features, scaler

def demonstrate_feature_extraction(dataset_path, output_dir='data'):
    """
    Demonstrate feature extraction from an apple disease image
    
    Args:
        dataset_path (str): Path to the dataset directory
        output_dir (str): Directory to save outputs
    """
    # Find an apple scab sample image
    apple_scab_dir = os.path.join(dataset_path, "apple", "apple scab")
    sample_image = os.path.join(apple_scab_dir, os.listdir(apple_scab_dir)[0])
    
    print(f"Demonstrating feature extraction on sample image: {sample_image}")
    
    # Extract features with visualization
    features, visualization = extract_features(sample_image, visualize=True)
    
    # Save and show visualization
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"feature_extraction_demo_{timestamp}.png")
    visualization.savefig(output_path)
    
    # Print feature information
    print("\nFeature vector dimensions:", features.shape)
    print("\nFeature vector:")
    feature_names = [
        "R-mean", "G-mean", "B-mean", 
        "R-std", "G-std", "B-std",
        "Contrast", "Energy", "Homogeneity", "Entropy",
        "Num Contours", "Contour Area", "Circularity", "Aspect Ratio"
    ]
    
    # Display features in a table format
    feature_df = pd.DataFrame({
        'Feature Name': feature_names,
        'Value': features
    })
    print(feature_df)
    
    print(f"\nVisualization saved to: {output_path}")
    
    # Return the features for further use if needed
    return features, sample_image

if __name__ == "__main__":
    # Path to the dataset
    dataset_path = "data/dataset/apple_disease"
    
    # Demonstrate feature extraction
    print("Starting feature extraction demonstration...")
    features, sample_image = demonstrate_feature_extraction(dataset_path)
    
    # Optionally, extract features from a small subset of the dataset
    print("\nExtracting features from a sample of the dataset...")
    features_array, labels, file_paths = load_dataset(dataset_path, max_samples=5)
    
    # Normalize features
    normalized_features, _ = normalize_features(features_array)
    
    # Show sample of the extracted dataset
    print(f"\nExtracted features from {len(features_array)} images")
    print(f"Features array shape: {features_array.shape}")
    print(f"Labels array shape: {labels.shape}")
    
    # Print sample data
    for i in range(min(3, len(features_array))):
        print(f"\nSample {i+1}:")
        print(f"  File: {file_paths[i]}")
        print(f"  Label: {labels[i]} ({'apple scab' if labels[i] == 1 else 'other'})")
        print(f"  First 5 features: {features_array[i][:5]}")
        print(f"  Normalized first 5 features: {normalized_features[i][:5]}")