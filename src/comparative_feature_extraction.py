import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
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

def extract_lesion_pattern_feature(image):
    """
    Extract feature representing lesion pattern (higher value = more regular/distinct pattern)
    
    Args:
        image (numpy.ndarray): Input image
        
    Returns:
        tuple: (feature value, visualization images)
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Find contours from edges
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create an image with the contours drawn
    contour_img = image.copy()
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 1)
    
    # Pattern regularity metrics
    if len(contours) == 0:
        return 0, [gray, edges, contour_img]
    
    # Calculate average contour area and perimeter
    avg_area = np.mean([cv2.contourArea(cnt) for cnt in contours if cv2.contourArea(cnt) > 5])
    avg_perimeter = np.mean([cv2.arcLength(cnt, True) for cnt in contours if cv2.contourArea(cnt) > 5])
    
    # Calculate circularity of contours (more circular = more regular)
    circularities = [4 * np.pi * cv2.contourArea(cnt) / (cv2.arcLength(cnt, True) ** 2) 
                     for cnt in contours if cv2.contourArea(cnt) > 5 and cv2.arcLength(cnt, True) > 0]
    
    # Normalized regularity score (higher = more regular/distinct)
    regularity = np.mean(circularities) if len(circularities) > 0 else 0
    
    # Scale to approximate range in original document (5-10)
    scaled_regularity = 5 + (regularity * 5)
    
    return min(max(scaled_regularity, 5), 10), [gray, edges, contour_img]

def extract_surface_texture_feature(image):
    """
    Extract feature representing surface texture (higher value = smoother texture)
    
    Args:
        image (numpy.ndarray): Input image
        
    Returns:
        tuple: (feature value, visualization images)
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Apply different texture detection techniques
    
    # 1. Laplacian for edge detection (highlights texture transitions)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian_normalized = cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # 2. Sobel filter (gradient magnitude for texture direction)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_magnitude = cv2.magnitude(sobelx, sobely)
    sobel_normalized = cv2.normalize(sobel_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # 3. Local Binary Pattern-like approach for texture
    # Simplified version - we threshold the blurred image to highlight texture patterns
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, texture_binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Calculate local variance as a measure of texture
    # Higher variance = rougher texture, lower variance = smoother texture
    kernel_size = 5
    local_variance = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    local_variance = cv2.Laplacian(local_variance, cv2.CV_64F).var()
    
    # Invert so higher value = smoother texture
    smoothness = 1.0 / (1.0 + local_variance)
    
    # Scale to approximate range in original document (4-9)
    scaled_smoothness = 4 + (smoothness * 5)
    
    return min(max(scaled_smoothness, 4), 9), [laplacian_normalized, sobel_normalized, texture_binary]

def extract_features(image_path, visualize=False):
    """
    Extract only lesion pattern and surface texture features from an image
    
    Args:
        image_path (str): Path to the image file
        visualize (bool): Whether to visualize the process
        
    Returns:
        tuple: (features, visualization if requested)
    """
    # Load and preprocess the image
    img = load_image(image_path)
    
    # Extract features
    lesion_pattern, lesion_viz = extract_lesion_pattern_feature(img)
    surface_texture, texture_viz = extract_surface_texture_feature(img)
    
    # Combine features
    features = np.array([lesion_pattern, surface_texture])
    
    # Create visualization if requested
    if visualize:
        fig = plt.figure(figsize=(15, 12))
        
        # Original image
        plt.subplot(3, 3, 1)
        plt.imshow(img)
        plt.title('Original Image')
        
        # Lesion pattern detection steps
        plt.subplot(3, 3, 2)
        plt.imshow(lesion_viz[0], cmap='gray')
        plt.title('Grayscale')
        
        plt.subplot(3, 3, 3)
        plt.imshow(lesion_viz[1], cmap='gray')
        plt.title('Edge Detection')
        
        plt.subplot(3, 3, 4)
        plt.imshow(lesion_viz[2])
        plt.title('Contours (Lesion Pattern)')
        
        # Surface texture detection steps
        plt.subplot(3, 3, 5)
        plt.imshow(texture_viz[0], cmap='gray')
        plt.title('Laplacian (Surface Texture)')
        
        plt.subplot(3, 3, 6)
        plt.imshow(texture_viz[1], cmap='gray')
        plt.title('Sobel Magnitude (Texture Direction)')
        
        plt.subplot(3, 3, 7)
        plt.imshow(texture_viz[2], cmap='gray')
        plt.title('Texture Binary Pattern')
        
        # Feature bar chart
        plt.subplot(3, 3, 8)
        plt.bar(['Lesion Pattern', 'Surface Texture'], features, color=['green', 'blue'])
        plt.ylim([0, 10])
        plt.title('Extracted Features')
        
        # Add image filename
        plt.subplot(3, 3, 9)
        plt.text(0.5, 0.5, os.path.basename(image_path), 
                 horizontalalignment='center', verticalalignment='center',
                 wrap=True, fontsize=8)
        plt.axis('off')
        
        plt.tight_layout()
        return features, fig
    
    return features

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

def compare_and_visualize_classes(dataset_path, output_dir='data', num_samples=10):
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
        "Lesion Pattern": features[:, 0],
        "Surface Texture": features[:, 1],
        "Class": class_labels,
        "File": file_paths
    })
    
    # Save to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(output_dir, f"comparative_features_{timestamp}.csv")
    df.to_csv(csv_path, index=False)
    
    print(f"\nExtracted features from {len(features)} images")
    print(f"CSV file saved to: {csv_path}")
    
    # Plot features by class
    plt.figure(figsize=(12, 10))
    
    # Scatter plot with different colors for each class
    for cls_name, color in zip(["apple scab", "healthy"], ["red", "green"]):
        cls_data = df[df["Class"] == cls_name]
        plt.scatter(
            cls_data["Lesion Pattern"], 
            cls_data["Surface Texture"],
            alpha=0.7, 
            label=cls_name,
            color=color
        )
    
    plt.xlabel('Lesion Pattern (higher = more regular/distinct)')
    plt.ylabel('Surface Texture (higher = smoother)')
    plt.title('Apple Disease Feature Comparison')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add text labels for each point
    for i, row in df.iterrows():
        plt.annotate(
            os.path.basename(row['File'])[:10] + '...',  # Shortened filename
            (row['Lesion Pattern'], row['Surface Texture']),
            fontsize=7,
            alpha=0.7,
            xytext=(5, 5),
            textcoords='offset points'
        )
    
    # Save the scatter plot
    plot_path = os.path.join(output_dir, f"comparative_features_scatter_{timestamp}.png")
    plt.savefig(plot_path)
    
    print(f"Feature comparison plot saved to: {plot_path}")
    
    # Create and save a visualization for one sample from each class
    print("\nGenerating detailed visualizations for sample images...")
    
    # Get sample images
    apple_scab_sample = df[df["Class"] == "apple scab"]["File"].iloc[0]
    healthy_sample = df[df["Class"] == "healthy"]["File"].iloc[0]
    
    # Generate visualizations
    _, apple_scab_viz = extract_features(apple_scab_sample, visualize=True)
    scab_viz_path = os.path.join(output_dir, f"apple_scab_features_viz_{timestamp}.png")
    apple_scab_viz.savefig(scab_viz_path)
    print(f"Apple scab visualization saved to: {scab_viz_path}")
    
    _, healthy_viz = extract_features(healthy_sample, visualize=True)
    healthy_viz_path = os.path.join(output_dir, f"healthy_features_viz_{timestamp}.png")
    healthy_viz.savefig(healthy_viz_path)
    print(f"Healthy apple visualization saved to: {healthy_viz_path}")
    
    return csv_path, plot_path, scab_viz_path, healthy_viz_path

if __name__ == "__main__":
    # Path to the dataset
    dataset_path = "data/dataset/apple_disease"
    
    # Compare 10 apple scab and 10 healthy images with enhanced visualizations
    print("Comparing apple scab and healthy classes with enhanced visualizations...")
    csv_path, plot_path, scab_viz_path, healthy_viz_path = compare_and_visualize_classes(dataset_path)
    
    print("\nComparison complete!")