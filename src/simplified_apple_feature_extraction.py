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
        float: Lesion pattern feature value
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Calculate the pattern regularity based on edge statistics
    # Higher values indicate more regular/distinct patterns
    
    # Find contours from edges
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # Pattern regularity metrics
    if len(contours) == 0:
        return 0
    
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
    
    return min(max(scaled_regularity, 5), 10)

def extract_surface_texture_feature(image):
    """
    Extract feature representing surface texture (higher value = smoother texture)
    
    Args:
        image (numpy.ndarray): Input image
        
    Returns:
        float: Surface texture feature value
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Calculate texture using GLCM properties
    # A simpler approach using local binary patterns (LBP)
    
    # Calculate local variance as a measure of texture
    # Higher variance = rougher texture, lower variance = smoother texture
    kernel_size = 5
    local_variance = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    local_variance = cv2.Laplacian(local_variance, cv2.CV_64F).var()
    
    # Invert so higher value = smoother texture
    smoothness = 1.0 / (1.0 + local_variance)
    
    # Scale to approximate range in original document (4-9)
    scaled_smoothness = 4 + (smoothness * 5)
    
    return min(max(scaled_smoothness, 4), 9)

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
    lesion_pattern = extract_lesion_pattern_feature(img)
    surface_texture = extract_surface_texture_feature(img)
    
    # Combine features
    features = np.array([lesion_pattern, surface_texture])
    
    # Create visualization if requested
    if visualize:
        # Original image
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        plt.imshow(img)
        plt.title('Original Image')
        
        # Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        plt.subplot(2, 2, 2)
        plt.imshow(gray, cmap='gray')
        plt.title('Grayscale')
        
        # Edge detection for lesion pattern
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        plt.subplot(2, 2, 3)
        plt.imshow(edges, cmap='gray')
        plt.title('Edge Detection (for Lesion Pattern)')
        
        # Feature bar chart
        plt.subplot(2, 2, 4)
        plt.bar(['Lesion Pattern', 'Surface Texture'], features)
        plt.title('Extracted Features')
        plt.ylim([0, 10])
        
        plt.tight_layout()
        return features, plt.gcf()
    
    return features

def load_apple_scab_dataset(dataset_path, max_samples=None):
    """
    Load only the apple scab class images and extract features
    
    Args:
        dataset_path (str): Path to the dataset directory
        max_samples (int): Maximum number of samples to load
        
    Returns:
        tuple: (features, file_paths)
    """
    features_list = []
    file_paths = []
    
    # Path to apple scab directory
    apple_scab_dir = os.path.join(dataset_path, "apple", "apple scab")
    
    # Get image files in the class directory
    image_files = [f for f in os.listdir(apple_scab_dir) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # Limit number of samples if specified
    if max_samples:
        image_files = image_files[:max_samples]
    
    print(f"Processing {len(image_files)} images from 'apple scab' class...")
    
    # Process each image
    for img_file in image_files:
        img_path = os.path.join(apple_scab_dir, img_file)
        try:
            # Extract features
            img_features = extract_features(img_path)
            
            # Store features and path
            features_list.append(img_features)
            file_paths.append(img_path)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    return np.array(features_list), file_paths

def demonstrate_feature_extraction(dataset_path, output_dir='data'):
    """
    Demonstrate feature extraction from an apple scab image
    
    Args:
        dataset_path (str): Path to the dataset directory
        output_dir (str): Directory to save outputs
    """
    # Find an apple scab sample image
    apple_scab_dir = os.path.join(dataset_path, "apple", "apple scab")
    sample_image = os.path.join(apple_scab_dir, os.listdir(apple_scab_dir)[0])
    
    print(f"Demonstrating simplified feature extraction on sample image: {sample_image}")
    
    # Extract features with visualization
    features, visualization = extract_features(sample_image, visualize=True)
    
    # Save and show visualization
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"simplified_feature_extraction_demo_{timestamp}.png")
    visualization.savefig(output_path)
    
    # Print feature information
    print("\nFeature vector dimensions:", features.shape)
    print("\nFeature vector:")
    feature_names = ["Lesion Pattern", "Surface Texture"]
    
    # Display features in a table format
    feature_df = pd.DataFrame({
        'Feature Name': feature_names,
        'Value': features
    })
    print(feature_df)
    
    print(f"\nVisualization saved to: {output_path}")
    
    # Return the features for further use if needed
    return features, sample_image

def create_dataset_csv(dataset_path, output_dir='data', max_samples=None):
    """
    Create a CSV file with extracted features from apple scab images
    
    Args:
        dataset_path (str): Path to the dataset directory
        output_dir (str): Directory to save the CSV file
        max_samples (int): Maximum number of samples to process
    """
    # Extract features from apple scab images
    features, file_paths = load_apple_scab_dataset(dataset_path, max_samples)
    
    # Create dataframe
    df = pd.DataFrame(features, columns=["Lesion Pattern", "Surface Texture"])
    
    # Add file paths and class label
    df["File"] = file_paths
    df["Class"] = "apple scab"
    
    # Save to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"apple_scab_features_{timestamp}.csv")
    df.to_csv(output_path, index=False)
    
    print(f"\nExtracted features from {len(features)} images")
    print(f"CSV file saved to: {output_path}")
    
    # Plot the feature distribution
    plt.figure(figsize=(10, 8))
    plt.scatter(features[:, 0], features[:, 1], alpha=0.7, label="Apple Scab")
    plt.xlabel('Lesion Pattern (higher = more regular/distinct)')
    plt.ylabel('Surface Texture (higher = smoother)')
    plt.title('Apple Scab Samples - Feature Distribution')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add some margin to the plot
    plt.xlim(min(features[:, 0]) - 0.5, max(features[:, 0]) + 0.5)
    plt.ylim(min(features[:, 1]) - 0.5, max(features[:, 1]) + 0.5)
    
    # Save the plot
    plot_path = os.path.join(output_dir, f"apple_scab_features_distribution_{timestamp}.png")
    plt.savefig(plot_path)
    
    print(f"Feature distribution plot saved to: {plot_path}")
    
    return output_path, plot_path

if __name__ == "__main__":
    # Path to the dataset
    dataset_path = "data/dataset/apple_disease"
    
    # Demonstrate feature extraction
    print("Starting simplified feature extraction demonstration...")
    features, sample_image = demonstrate_feature_extraction(dataset_path)
    
    # Create a CSV file with all apple scab images (limit to 50 for demonstration)
    print("\nCreating dataset with simplified features...")
    csv_path, plot_path = create_dataset_csv(dataset_path, max_samples=50)
    
    print("\nDemonstration complete!")