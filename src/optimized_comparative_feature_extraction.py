import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage import color, exposure

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
    Extract color-based features that may help identify disease symptoms
    
    Args:
        image (numpy.ndarray): Input RGB image
        
    Returns:
        tuple: (feature values, visualization images)
    """
    # Convert to different color spaces
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    
    # Extract color channels
    h, s, v = cv2.split(hsv)
    l, a, b = cv2.split(lab)
    
    # Calculate color statistics
    # Green vs brown/yellow ratio (apple scab shows brown spots)
    g_channel = image[:,:,1]
    r_channel = image[:,:,0]
    
    # Calculate color ratios for disease detection
    # Apple scab typically shows as dark brown/black spots
    # Calculate the ratio of green pixels to brownish/dark pixels
    green_mask = (g_channel > 100) & (g_channel > r_channel)
    dark_mask = (v < 100)
    
    green_pixel_ratio = np.sum(green_mask) / (image.shape[0] * image.shape[1])
    dark_pixel_ratio = np.sum(dark_mask) / (image.shape[0] * image.shape[1])
    
    # Calculate color variability in a and b channels (Lab color space)
    # Higher variability in a channel often indicates lesions
    a_std = np.std(a)
    b_std = np.std(b)
    
    # Combine features
    color_disease_indicator = dark_pixel_ratio / (green_pixel_ratio + 0.01)
    color_contrast_feature = (a_std + b_std) / 2
    
    # Feature values
    color_features = [
        color_disease_indicator * 10,  # Scale for better visualization
        color_contrast_feature / 5     # Normalize to similar scale
    ]
    
    # Visualization images
    visualizations = [
        hsv,
        cv2.merge([a, a, a]),  # Visualize 'a' channel (helps show disease spots)
        cv2.merge([np.uint8(dark_mask*255), np.uint8(green_mask*255), np.zeros_like(dark_mask, dtype=np.uint8)])
    ]
    
    return color_features, visualizations

def extract_advanced_texture_feature(image):
    """
    Extract advanced texture features using GLCM and LBP
    
    Args:
        image (numpy.ndarray): Input image
        
    Returns:
        tuple: (feature values, visualization images)
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # 1. Gray Level Co-occurrence Matrix (GLCM) features
    # Quantize the image to fewer gray levels to make GLCM computation feasible
    gray_quantized = np.uint8(gray / 16) * 16
    
    # Calculate GLCM with multiple angles and distances
    distances = [1, 3]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = graycomatrix(
        gray_quantized, 
        distances=distances, 
        angles=angles, 
        symmetric=True, 
        normed=True
    )
    
    # Calculate GLCM properties
    contrast = np.mean(graycoprops(glcm, 'contrast'))
    dissimilarity = np.mean(graycoprops(glcm, 'dissimilarity'))
    homogeneity = np.mean(graycoprops(glcm, 'homogeneity'))
    energy = np.mean(graycoprops(glcm, 'energy'))
    correlation = np.mean(graycoprops(glcm, 'correlation'))
    
    # 2. Local Binary Pattern
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    
    # Get histogram of LBP
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    
    # Calculate LBP statistics
    lbp_entropy = -np.sum(hist * np.log2(hist + 1e-10))  # Entropy of LBP distribution
    lbp_mean = np.mean(lbp)
    lbp_var = np.var(lbp)
    
    # Define two texture features that should help differentiate healthy vs diseased apples
    # Feature 1: Based on homogeneity and contrast (healthy is more homogeneous, less contrast)
    texture_smoothness = (homogeneity - contrast + 1) / 2
    
    # Feature 2: Based on LBP statistics (diseased apples have more varied textures)
    # Low entropy and variance typically means more uniform texture (healthy apple)
    texture_regularity = 1 - (lbp_entropy / 10)
    
    # Scale features to have interpretable ranges
    smoothness_scaled = texture_smoothness * 10
    regularity_scaled = texture_regularity * 10
    
    # For visualization
    lbp_image = np.uint8((lbp / lbp.max()) * 255)
    contrast_image = exposure.equalize_hist(gray) * 255
    
    # Return texture features and visualizations
    texture_features = [smoothness_scaled, regularity_scaled]
    visualizations = [lbp_image, contrast_image, gray]
    
    return texture_features, visualizations

def extract_lesion_pattern_feature(image):
    """
    Extract feature representing lesion pattern with improvements
    
    Args:
        image (numpy.ndarray): Input image
        
    Returns:
        tuple: (feature values, visualization images)
    """
    # Convert to LAB color space (better for detecting lesions)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Enhanced preprocessing for better lesion detection
    # Apply CLAHE to improve contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l)
    
    # Combine the a channel (useful for detecting lesions) with enhanced L
    a_stretched = cv2.normalize(a, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Apply bilateral filter to preserve edges while reducing noise
    filtered = cv2.bilateralFilter(a_stretched, 9, 75, 75)
    
    # Apply Otsu's thresholding to find potential lesion regions
    _, binary = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Morphological operations to refine the lesion regions
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Find contours in the binary image
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create visualization image
    contour_img = image.copy()
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 1)
    
    # Calculate lesion features
    if len(contours) == 0:
        lesion_density = 0
        lesion_distribution = 0
    else:
        # Filter out very small contours
        valid_contours = [c for c in contours if cv2.contourArea(c) > 10]
        
        if not valid_contours:
            lesion_density = 0
            lesion_distribution = 0
        else:
            # Calculate lesion density (ratio of lesion area to total area)
            total_area = image.shape[0] * image.shape[1]
            lesion_area = sum(cv2.contourArea(c) for c in valid_contours)
            lesion_density = lesion_area / total_area
            
            # Calculate lesion distribution pattern
            # Get centroids of all contours
            centroids = np.array([np.mean(c.reshape(-1, 2), axis=0) for c in valid_contours])
            
            if len(centroids) > 1:
                # Calculate distances between all pairs of centroids
                from scipy.spatial.distance import pdist
                distances = pdist(centroids)
                
                # Calculate coefficient of variation of distances
                # Low CV means regular pattern, high CV means clustered/irregular
                cv_distances = np.std(distances) / (np.mean(distances) + 1e-10)
                
                # Convert to a feature where higher means more regular pattern
                lesion_distribution = 1 - min(cv_distances, 1)
            else:
                lesion_distribution = 0.5  # Single lesion case
    
    # Scale features
    lesion_density_scaled = lesion_density * 20  # Scale to approximate 0-10 range
    lesion_distribution_scaled = lesion_distribution * 10
    
    # Return lesion features and visualizations
    lesion_features = [lesion_density_scaled, lesion_distribution_scaled]
    visualizations = [enhanced_l, a_stretched, opening, contour_img]
    
    return lesion_features, visualizations

def extract_features(image_path, visualize=False):
    """
    Extract optimized features from an image
    
    Args:
        image_path (str): Path to the image file
        visualize (bool): Whether to visualize the process
        
    Returns:
        tuple: (features, visualization if requested)
    """
    # Load and preprocess the image
    img = load_image(image_path)
    
    # Extract features
    color_features, color_viz = extract_color_features(img)
    texture_features, texture_viz = extract_advanced_texture_feature(img)
    lesion_features, lesion_viz = extract_lesion_pattern_feature(img)
    
    # Combine features for the most useful ones - only keeping Texture Regularity and Lesion Density
    selected_features = np.array([
        texture_features[1],    # Texture regularity
        lesion_features[0]      # Lesion density
    ])
    
    # Create visualization if requested
    if visualize:
        fig = plt.figure(figsize=(15, 12))
        
        # Original image
        plt.subplot(4, 3, 1)
        plt.imshow(img)
        plt.title('Original Image')
        
        # Color feature visualization
        plt.subplot(4, 3, 2)
        plt.imshow(color_viz[0])
        plt.title('HSV Color Space')
        
        plt.subplot(4, 3, 3)
        plt.imshow(color_viz[1], cmap='viridis')
        plt.title('A Channel (Lab)')
        
        plt.subplot(4, 3, 4)
        plt.imshow(color_viz[2])
        plt.title('Green/Dark Mask')
        
        # Texture feature visualization
        plt.subplot(4, 3, 5)
        plt.imshow(texture_viz[0], cmap='gray')
        plt.title('Local Binary Pattern')
        
        plt.subplot(4, 3, 6)
        plt.imshow(texture_viz[1], cmap='gray')
        plt.title('Contrast Enhanced')
        
        plt.subplot(4, 3, 7)
        plt.imshow(texture_viz[2], cmap='gray')
        plt.title('Grayscale')
        
        # Lesion pattern visualization
        plt.subplot(4, 3, 8)
        plt.imshow(lesion_viz[0], cmap='gray')
        plt.title('Enhanced L (Lab)')
        
        plt.subplot(4, 3, 9)
        plt.imshow(lesion_viz[1], cmap='gray')
        plt.title('A Channel Stretched')
        
        plt.subplot(4, 3, 10)
        plt.imshow(lesion_viz[2], cmap='gray')
        plt.title('Lesion Binary Mask')
        
        plt.subplot(4, 3, 11)
        plt.imshow(lesion_viz[3])
        plt.title('Detected Lesions')
        
        # Feature bar chart
        plt.subplot(4, 3, 12)
        plt.bar(['Texture Regularity', 'Lesion Density'], 
                selected_features, color=['blue', 'green'])
        plt.ylim([0, 12])
        plt.title('Selected Features')
        
        plt.tight_layout()
        return selected_features, fig
    
    return selected_features

def load_balanced_dataset(dataset_path, num_samples=100):
    """
    Load a balanced dataset from apple scab and healthy classes
    
    Args:
        dataset_path (str): Path to the dataset directory
        num_samples (int): Number of samples to load from each class
        
    Returns:
        tuple: (features, labels, file_paths)
    """
    features_list = []
    labels = []
    file_paths = []
    
    # Path to apple scab and healthy directories
    apple_scab_dir = os.path.join(dataset_path, "apple", "apple scab")
    healthy_dir = os.path.join(dataset_path, "apple", "healthy")
    
    # Process apple scab images
    scab_image_files = [f for f in os.listdir(apple_scab_dir) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:num_samples]
    
    print(f"Processing {len(scab_image_files)} images from 'apple scab' class...")
    for img_file in scab_image_files:
        img_path = os.path.join(apple_scab_dir, img_file)
        try:
            # Extract features
            img_features = extract_features(img_path)
            
            # Store features, label, and path
            features_list.append(img_features)
            labels.append(1)  # 1 for apple scab
            file_paths.append(img_path)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    # Process healthy images
    healthy_image_files = [f for f in os.listdir(healthy_dir)
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:num_samples]
    
    print(f"Processing {len(healthy_image_files)} images from 'healthy' class...")
    for img_file in healthy_image_files:
        img_path = os.path.join(healthy_dir, img_file)
        try:
            # Extract features
            img_features = extract_features(img_path)
            
            # Store features, label, and path
            features_list.append(img_features)
            labels.append(0)  # 0 for healthy
            file_paths.append(img_path)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    return np.array(features_list), np.array(labels), file_paths

def compare_and_visualize_classes(dataset_path, output_dir='data/feature_extraction_results', num_samples=10):
    """
    Compare features between apple scab and healthy classes with visualizations
    
    Args:
        dataset_path (str): Path to the dataset directory
        output_dir (str): Directory to save outputs
        num_samples (int): Number of samples per class
    """
    # Load balanced dataset
    features, labels, file_paths = load_balanced_dataset(dataset_path, num_samples)
    
    # Create dataframe
    class_labels = ["healthy" if label == 0 else "apple scab" for label in labels]
    df = pd.DataFrame({
        "Texture Regularity": features[:, 0],
        "Lesion Density": features[:, 1],
        "Class": class_labels,
        "File": file_paths
    })
    
    # Create timestamp and run-specific directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Save to CSV
    csv_path = os.path.join(run_dir, "features.csv")
    df.to_csv(csv_path, index=False)
    
    print(f"\nExtracted optimized features from {len(features)} images")
    print(f"CSV file saved to: {csv_path}")
    
    # Create scatter plot to visualize features
    plt.figure(figsize=(10, 8))
    
    # Plot Texture Regularity vs Lesion Density
    for cls_name, color in zip(["apple scab", "healthy"], ["red", "green"]):
        cls_data = df[df["Class"] == cls_name]
        plt.scatter(
            cls_data["Texture Regularity"], 
            cls_data["Lesion Density"],
            alpha=0.7, 
            label=cls_name,
            color=color
        )
    
    plt.xlabel('Texture Regularity')
    plt.ylabel('Lesion Density')
    plt.title('Texture Regularity vs Lesion Density')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add text labels for each point
    for i, row in df.iterrows():
        plt.annotate(
            os.path.basename(row['File'])[:8] + '...',  # Shortened filename
            (row['Texture Regularity'], row['Lesion Density']),
            fontsize=7,
            alpha=0.7,
            xytext=(5, 5),
            textcoords='offset points'
        )
    
    plt.tight_layout()
    
    # Save the scatter plot
    plot_path = os.path.join(run_dir, "scatter_plot.png")
    plt.savefig(plot_path)
    
    print(f"Feature comparison plot saved to: {plot_path}")
    
    # Create and save a visualization for one sample from each class
    print("\nGenerating detailed visualizations for sample images...")
    
    # Get sample images
    apple_scab_sample = df[df["Class"] == "apple scab"]["File"].iloc[0]
    healthy_sample = df[df["Class"] == "healthy"]["File"].iloc[0]
    
    # Generate visualizations
    _, apple_scab_viz = extract_features(apple_scab_sample, visualize=True)
    scab_viz_path = os.path.join(run_dir, "apple_scab_visualization.png")
    apple_scab_viz.savefig(scab_viz_path)
    print(f"Apple scab visualization saved to: {scab_viz_path}")
    
    _, healthy_viz = extract_features(healthy_sample, visualize=True)
    healthy_viz_path = os.path.join(run_dir, "healthy_visualization.png")
    healthy_viz.savefig(healthy_viz_path)
    print(f"Healthy apple visualization saved to: {healthy_viz_path}")
    
    # Create a README file with run information
    readme_path = os.path.join(run_dir, "README.txt")
    with open(readme_path, 'w') as f:
        f.write(f"Feature Extraction Run: {timestamp}\n")
        f.write(f"Number of samples: {num_samples} per class\n")
        f.write("Features extracted: Texture Regularity, Lesion Density\n\n")
        f.write("Files in this directory:\n")
        f.write("- features.csv: CSV file containing extracted features for all samples\n")
        f.write("- scatter_plot.png: Visualization of Texture Regularity vs Lesion Density\n")
        f.write("- apple_scab_visualization.png: Detailed visualization of an apple scab sample\n")
        f.write("- healthy_visualization.png: Detailed visualization of a healthy sample\n")
        f.write("- README.txt: This file\n")
    
    print(f"Run summary saved to: {readme_path}")
    
    return csv_path, plot_path, scab_viz_path, healthy_viz_path

if __name__ == "__main__":
    # Path to the dataset
    dataset_path = "data/dataset/apple_disease"
    
    # Compare apple scab and healthy images with optimized feature extraction
    print("Comparing apple scab and healthy classes with optimized feature extraction...")
    csv_path, plot_path, scab_viz_path, healthy_viz_path = compare_and_visualize_classes(dataset_path)
    
    print("\nOptimized comparison complete!")