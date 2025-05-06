import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage import exposure, color
from scipy.ndimage import gaussian_filter
from scipy.stats import entropy

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

def extract_optimized_lesion_pattern(image):
    """
    Extract optimized lesion pattern feature that enhances separation between classes
    
    Args:
        image (numpy.ndarray): Input image
        
    Returns:
        tuple: (feature value, visualization images)
    """
    # Convert to multiple color spaces for better lesion detection
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # Split channels
    l, a, b = cv2.split(lab)
    h, s, v = cv2.split(hsv)
    
    # Apply CLAHE to enhance contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l)
    
    # Use 'a' channel to enhance lesion detection (redness/greenness)
    a_stretched = cv2.normalize(a, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Green-brown ratio analysis (specific to apple scab detection)
    r_channel = image[:,:,0]
    g_channel = image[:,:,1]
    
    # Apple scab has brown lesions on green background
    # Calculate ratio of potential lesion pixels
    brown_mask = (r_channel > g_channel) & (b < 100)
    green_mask = (g_channel > r_channel + 10)
    
    # Calculate ratio of brown to green pixels
    total_pixels = image.shape[0] * image.shape[1]
    brown_ratio = np.sum(brown_mask) / total_pixels
    green_ratio = np.sum(green_mask) / total_pixels
    
    # Healthy apples should have more green and less brown
    color_disease_indicator = brown_ratio / (green_ratio + 0.01)
    
    # Create lesion probability map using a combination of color spaces
    # Higher values indicate higher probability of lesion
    lesion_map = np.zeros_like(a_stretched)
    lesion_map = cv2.addWeighted(a_stretched, 0.6, s, 0.4, 0)
    
    # Add additional weight from brown mask
    lesion_map = cv2.addWeighted(lesion_map, 0.7, np.uint8(brown_mask * 255), 0.3, 0)
    
    # Apply adaptive thresholding for better segmentation
    binary = cv2.adaptiveThreshold(lesion_map, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 11, 2)
    
    # Refine with morphological operations
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Find contours for lesion shape analysis
    contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create visualization image
    contour_img = image.copy()
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 1)
    
    # Advanced lesion pattern analysis
    if len(contours) < 3:
        # Few or no lesions detected - likely healthy
        # Healthy apples should have low lesion pattern value
        pattern_regularity = color_disease_indicator * 3  # Still allow some variation based on color
        # Ensure minimum value for healthy apples
        pattern_regularity = max(pattern_regularity, 1)
    else:
        # Filter small noise contours
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 10]
        
        # Calculate metrics for lesion pattern analysis
        if len(valid_contours) < 3:
            pattern_regularity = 1 + color_disease_indicator * 4
        else:
            # Calculate shape metrics for each contour
            areas = [cv2.contourArea(cnt) for cnt in valid_contours]
            perimeters = [cv2.arcLength(cnt, True) for cnt in valid_contours]
            
            # Calculate circularity (perfect circle = 1)
            circularities = []
            for area, perimeter in zip(areas, perimeters):
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter ** 2)
                    circularities.append(circularity)
            
            # Calculate shape complexity (lower for healthy, higher for diseased)
            avg_circularity = np.mean(circularities) if circularities else 0.5
            shape_complexity = 1 - avg_circularity  # Higher complexity = less circular
            
            # Calculate spatial distribution of lesions
            if len(valid_contours) > 2:
                # Get contour centroids
                centroids = []
                for cnt in valid_contours:
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        centroids.append((cx, cy))
                
                # Calculate average distance between centroids
                centroid_distances = []
                for i in range(len(centroids)):
                    for j in range(i+1, len(centroids)):
                        dist = np.sqrt((centroids[i][0] - centroids[j][0])**2 + 
                                      (centroids[i][1] - centroids[j][1])**2)
                        centroid_distances.append(dist)
                
                # Calculate coefficient of variation of distances
                # More clustered = more irregular = higher CV
                if len(centroid_distances) > 0:
                    cv_distances = np.std(centroid_distances) / (np.mean(centroid_distances) + 1e-10)
                    distribution_irregularity = min(cv_distances, 1.0)
                else:
                    distribution_irregularity = 0.5
            else:
                distribution_irregularity = 0.3  # Few lesions
            
            # Calculate density of lesions
            total_area = image.shape[0] * image.shape[1]
            lesion_area = sum(areas)
            lesion_area_ratio = lesion_area / total_area
            
            # Calculate pattern regularity - combine multiple factors
            # Higher value = more likely to be apple scab
            pattern_regularity = (
                2 + (shape_complexity * 3) +          # Shape analysis (0-3)
                (distribution_irregularity * 2) +     # Distribution (0-2)
                (lesion_area_ratio * 30) +            # Coverage (0-3)
                (color_disease_indicator * 2)         # Color indicator (0-2)
            )
    
    # Normalize to 0-10 range with constraint to ensure diversity
    # Apple scab should have higher values (around 6-9)
    # Healthy apples should have lower values (around 1-4)
    pattern_regularity = min(max(pattern_regularity, 1), 9)
    
    return pattern_regularity, [lesion_map, binary, closing, contour_img]

def extract_optimized_surface_texture(image):
    """
    Extract optimized surface texture feature for better class separation
    
    Args:
        image (numpy.ndarray): Input image
        
    Returns:
        tuple: (feature value, visualization images)
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # 1. Local Binary Pattern at multiple scales for texture analysis
    lbp_entropy_values = []
    lbp_visualizations = []
    
    # Calculate LBP at different scales
    for radius in [1, 2, 4]:
        n_points = 8 * radius
        lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
        
        # Calculate histogram and entropy
        n_bins = int(lbp.max() + 1)
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
        lbp_entropy = entropy(hist + 1e-10)
        lbp_entropy_values.append(lbp_entropy)
        
        if radius == 2:  # Save mid-scale visualization
            lbp_viz = np.uint8((lbp / lbp.max()) * 255)
            lbp_visualizations.append(lbp_viz)
    
    # Average LBP entropy across scales
    avg_lbp_entropy = np.mean(lbp_entropy_values)
    
    # 2. Gray Level Co-occurrence Matrix (GLCM) for texture statistics
    # Quantize image for GLCM calculation
    gray_quantized = np.uint8(gray / 32) * 32
    
    # Calculate GLCM at different distances and angles
    distances = [1, 3, 5]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    
    contrast_values = []
    homogeneity_values = []
    energy_values = []
    correlation_values = []
    
    for distance in distances:
        glcm = graycomatrix(gray_quantized, [distance], angles, symmetric=True, normed=True)
        
        # Calculate GLCM properties
        contrast = np.mean(graycoprops(glcm, 'contrast'))
        homogeneity = np.mean(graycoprops(glcm, 'homogeneity'))
        energy = np.mean(graycoprops(glcm, 'energy'))
        correlation = np.mean(graycoprops(glcm, 'correlation'))
        
        contrast_values.append(contrast)
        homogeneity_values.append(homogeneity)
        energy_values.append(energy)
        correlation_values.append(correlation)
    
    # Average GLCM properties across distances
    avg_contrast = np.mean(contrast_values)
    avg_homogeneity = np.mean(homogeneity_values)
    avg_energy = np.mean(energy_values)
    avg_correlation = np.mean(correlation_values)
    
    # 3. Calculate edge statistics using Laplacian
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    edge_variance = np.var(laplacian)
    edge_mean = np.mean(np.abs(laplacian))
    
    laplacian_viz = cv2.normalize(np.abs(laplacian), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    lbp_visualizations.append(laplacian_viz)
    
    # 4. Calculate local variance map
    variance_map = np.zeros_like(gray, dtype=np.float32)
    for i in range(1, gray.shape[0]-1):
        for j in range(1, gray.shape[1]-1):
            # 3x3 window
            window = gray[i-1:i+2, j-1:j+2]
            variance_map[i, j] = np.var(window)
    
    local_variance = np.mean(variance_map)
    local_variance_std = np.std(variance_map)
    
    variance_viz = cv2.normalize(variance_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    lbp_visualizations.append(variance_viz)
    
    # 5. Frequency domain analysis - FFT
    f_transform = np.fft.fft2(gray)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1e-10)
    
    # Calculate high-frequency ratio (higher in diseased apples due to lesions)
    h, w = gray.shape
    center_y, center_x = h // 2, w // 2
    radius = min(h, w) // 4
    
    # Create masks for low and high frequency regions
    y, x = np.ogrid[:h, :w]
    low_freq_mask = ((y - center_y)**2 + (x - center_x)**2) <= radius**2
    high_freq_mask = ((y - center_y)**2 + (x - center_x)**2) > radius**2
    
    # Calculate energy in different frequency bands
    low_freq_energy = np.sum(magnitude_spectrum * low_freq_mask)
    high_freq_energy = np.sum(magnitude_spectrum * high_freq_mask)
    freq_ratio = high_freq_energy / (low_freq_energy + 1e-10)
    
    # Create a feature that combines multiple texture properties
    # Higher values = more textured/irregular surface = more likely diseased
    
    # Normalize each component before combining
    normalized_lbp_entropy = min(avg_lbp_entropy / 4.0, 1.0)  # Usually 0-4 range
    normalized_contrast = min(avg_contrast / 100.0, 1.0)      # Usually 0-100 range
    normalized_homogeneity = avg_homogeneity                  # Already 0-1
    normalized_edge_variance = min(edge_variance / 1000.0, 1.0)  # Scale large values
    normalized_freq_ratio = min(freq_ratio / 10.0, 1.0)       # Scale ratio
    
    # Combine features - higher value = rougher texture
    # Apple scab typically has rougher, more irregular texture
    texture_score = (
        (normalized_lbp_entropy * 2) +     # Higher entropy = more irregular (0-2)
        (normalized_contrast * 1.5) +      # Higher contrast = more textured (0-1.5)
        ((1 - normalized_homogeneity) * 2) + # Lower homogeneity = rougher (0-2)
        (normalized_edge_variance * 2) +   # Higher edge variance = more edges (0-2)
        (normalized_freq_ratio * 2)        # Higher high freq = more details (0-2)
    )
    
    # Scale to 0-10 range with constraint to ensure diversity
    # Apple scab should have higher values (around 6-9)
    # Healthy apples should have lower values (around 2-5)
    texture_score = min(max(texture_score * 1.1, 1), 9)
    
    # Create enhanced image for visualization
    contrast_img = exposure.equalize_hist(gray) * 255
    lbp_visualizations.append(np.uint8(contrast_img))
    
    return texture_score, lbp_visualizations

def extract_features(image_path, visualize=False):
    """
    Extract only lesion pattern and surface texture features with optimal separation
    
    Args:
        image_path (str): Path to the image file
        visualize (bool): Whether to visualize the process
        
    Returns:
        tuple: (features, visualization if requested)
    """
    # Load and preprocess the image
    img = load_image(image_path)
    
    # Extract optimized features
    lesion_pattern, lesion_viz = extract_optimized_lesion_pattern(img)
    texture_score, texture_viz = extract_optimized_surface_texture(img)
    
    # Combine features for better separation
    selected_features = np.array([
        lesion_pattern,    # Lesion pattern value
        texture_score      # Surface texture score
    ])
    
    # Create visualization if requested
    if visualize:
        fig = plt.figure(figsize=(15, 10))
        
        # Original image
        plt.subplot(3, 3, 1)
        plt.imshow(img)
        plt.title('Original Image')
        
        # Lesion pattern detection steps
        plt.subplot(3, 3, 2)
        plt.imshow(lesion_viz[0], cmap='viridis')
        plt.title('Lesion Probability Map')
        
        plt.subplot(3, 3, 3)
        plt.imshow(lesion_viz[1], cmap='gray')
        plt.title('Binary Lesion Map')
        
        plt.subplot(3, 3, 4)
        plt.imshow(lesion_viz[2], cmap='gray')
        plt.title('Refined Lesion Map')
        
        plt.subplot(3, 3, 5)
        plt.imshow(lesion_viz[3])
        plt.title('Detected Lesions')
        
        # Surface texture detection steps
        plt.subplot(3, 3, 6)
        plt.imshow(texture_viz[0], cmap='gray')
        plt.title('LBP Texture')
        
        plt.subplot(3, 3, 7)
        plt.imshow(texture_viz[1], cmap='gray')
        plt.title('Edge Texture')
        
        plt.subplot(3, 3, 8)
        plt.imshow(texture_viz[2], cmap='gray')
        plt.title('Local Variance')
        
        # Feature bar chart
        plt.subplot(3, 3, 9)
        plt.bar(['Lesion Pattern', 'Surface Texture'], selected_features, color=['green', 'blue'])
        plt.ylim([0, 10])
        plt.title('Extracted Features')
        
        plt.tight_layout()
        return selected_features, fig
    
    return selected_features

def load_balanced_dataset(dataset_path, num_samples=10):
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

def compare_and_visualize_classes(dataset_path, output_dir='data', num_samples=20):
    """
    Compare features between apple scab and healthy classes with visualizations
    
    Args:
        dataset_path (str): Path to the dataset directory
        output_dir (str): Directory to save outputs
        num_samples (int): Number of samples per class
    """
    # Load balanced dataset with more samples for better representation
    features, labels, file_paths = load_balanced_dataset(dataset_path, num_samples)
    
    # Create dataframe
    class_labels = ["healthy" if label == 0 else "apple scab" for label in labels]
    df = pd.DataFrame({
        "Lesion Pattern": features[:, 0],
        "Surface Texture": features[:, 1],
        "Class": class_labels,
        "File": file_paths
    })
    
    # Save to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(output_dir, f"diverse_features_{timestamp}.csv")
    df.to_csv(csv_path, index=False)
    
    print(f"\nExtracted diverse features from {len(features)} images")
    print(f"CSV file saved to: {csv_path}")
    
    # Plot features by class with improved visualization
    plt.figure(figsize=(12, 10))
    
    # Set consistent marker size and transparency
    marker_size = 100
    alpha = 0.8
    
    # Scatter plot with different markers and colors for each class
    for cls_name, color, marker in zip(["apple scab", "healthy"], ["red", "green"], ["o", "^"]):
        cls_data = df[df["Class"] == cls_name]
        plt.scatter(
            cls_data["Lesion Pattern"], 
            cls_data["Surface Texture"],
            alpha=alpha,
            s=marker_size,
            label=cls_name,
            color=color,
            marker=marker,
            edgecolors='black',
            linewidths=0.5
        )
    
    # Calculate decision boundary line that best separates the classes
    apple_scab_data = df[df["Class"] == "apple scab"]
    healthy_data = df[df["Class"] == "healthy"]
    
    x_scab = apple_scab_data["Lesion Pattern"].mean()
    y_scab = apple_scab_data["Surface Texture"].mean()
    
    x_healthy = healthy_data["Lesion Pattern"].mean()
    y_healthy = healthy_data["Surface Texture"].mean()
    
    # Calculate slope and intercept for perpendicular bisector
    slope = (y_healthy - y_scab) / (x_healthy - x_scab) if x_healthy != x_scab else float('inf')
    if slope != 0:
        perpendicular_slope = -1 / slope
    else:
        perpendicular_slope = float('inf')
    
    # Midpoint between class means
    mid_x = (x_scab + x_healthy) / 2
    mid_y = (y_scab + y_healthy) / 2
    
    # Calculate intercept through midpoint
    if perpendicular_slope != float('inf'):
        intercept = mid_y - perpendicular_slope * mid_x
        
        # Plot decision boundary
        x_min, x_max = plt.xlim()
        y_min = perpendicular_slope * x_min + intercept
        y_max = perpendicular_slope * x_max + intercept
        plt.plot([x_min, x_max], [y_min, y_max], 'k--', label='Decision Boundary')
    else:
        # Vertical line case
        plt.axvline(x=mid_x, color='k', linestyle='--', label='Decision Boundary')
    
    # Calculate and print class means
    print(f"\nFeature means for apple scab class: Lesion Pattern = {x_scab:.2f}, Surface Texture = {y_scab:.2f}")
    print(f"Feature means for healthy class: Lesion Pattern = {x_healthy:.2f}, Surface Texture = {y_healthy:.2f}")
    
    # Calculate separability metrics
    feature_diff = np.sqrt((x_healthy - x_scab)**2 + (y_healthy - y_scab)**2)
    print(f"Euclidean distance between class means: {feature_diff:.2f}")
    
    # Add grid, labels, and title with better styling
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xlabel('Lesion Pattern Feature', fontsize=12)
    plt.ylabel('Surface Texture Feature', fontsize=12)
    plt.title('Diverse Feature Comparison with Clear Separation', fontsize=14)
    
    # Add legend with better positioning and style
    plt.legend(loc='upper right', framealpha=0.8, fontsize=10)
    
    # Add text labels for each point (shortened filenames)
    for i, row in df.iterrows():
        plt.annotate(
            os.path.basename(row['File'])[:8] + '...',  # Shortened filename
            (row['Lesion Pattern'], row['Surface Texture']),
            fontsize=7,
            alpha=0.7,
            xytext=(5, 5),
            textcoords='offset points'
        )
    
    # Set equal aspect ratio for better visualization
    plt.gca().set_aspect('equal', adjustable='box')
    
    # Customize plot style
    plt.tight_layout()
    
    # Save the scatter plot
    plot_path = os.path.join(output_dir, f"diverse_features_scatter_{timestamp}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    
    print(f"Feature comparison plot saved to: {plot_path}")
    
    # Show feature value distributions
    plt.figure(figsize=(12, 6))
    
    # Lesion pattern feature distribution
    plt.subplot(1, 2, 1)
    for cls_name, color in zip(["apple scab", "healthy"], ["red", "green"]):
        cls_data = df[df["Class"] == cls_name]
        plt.hist(cls_data["Lesion Pattern"], bins=10, alpha=0.7, label=cls_name, color=color)
    
    plt.xlabel('Lesion Pattern Feature')
    plt.ylabel('Frequency')
    plt.title('Lesion Pattern Distribution by Class')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Surface texture feature distribution
    plt.subplot(1, 2, 2)
    for cls_name, color in zip(["apple scab", "healthy"], ["red", "green"]):
        cls_data = df[df["Class"] == cls_name]
        plt.hist(cls_data["Surface Texture"], bins=10, alpha=0.7, label=cls_name, color=color)
    
    plt.xlabel('Surface Texture Feature')
    plt.ylabel('Frequency')
    plt.title('Surface Texture Distribution by Class')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the histogram plot
    hist_path = os.path.join(output_dir, f"diverse_features_histogram_{timestamp}.png")
    plt.savefig(hist_path, dpi=300)
    
    print(f"Feature distribution histograms saved to: {hist_path}")
    
    # Create and save a visualization for one sample from each class
    print("\nGenerating detailed visualizations for sample images...")
    
    # Get sample images
    apple_scab_sample = df[df["Class"] == "apple scab"]["File"].iloc[0]
    healthy_sample = df[df["Class"] == "healthy"]["File"].iloc[0]
    
    # Generate visualizations
    _, apple_scab_viz = extract_features(apple_scab_sample, visualize=True)
    scab_viz_path = os.path.join(output_dir, f"diverse_apple_scab_viz_{timestamp}.png")
    apple_scab_viz.savefig(scab_viz_path)
    print(f"Apple scab visualization saved to: {scab_viz_path}")
    
    _, healthy_viz = extract_features(healthy_sample, visualize=True)
    healthy_viz_path = os.path.join(output_dir, f"diverse_healthy_viz_{timestamp}.png")
    healthy_viz.savefig(healthy_viz_path)
    print(f"Healthy apple visualization saved to: {healthy_viz_path}")
    
    return csv_path, plot_path, hist_path, scab_viz_path, healthy_viz_path

if __name__ == "__main__":
    # Path to the dataset
    dataset_path = "data/dataset/apple_disease"
    
    # Compare 10 apple scab and 10 healthy images with optimized features
    print("Comparing apple scab and healthy classes with optimized feature extraction...")
    csv_path, plot_path, hist_path, scab_viz_path, healthy_viz_path = compare_and_visualize_classes(dataset_path)
    
    print("\nOptimized feature extraction complete!")