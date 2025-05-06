import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from svm import SVM
import pandas as pd
from tabulate import tabulate
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Use absolute paths to ensure images are saved correctly regardless of working directory
# Get the absolute path to the project directory
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')

# Create data directory if it doesn't exist
os.makedirs(DATA_DIR, exist_ok=True)

# Static data for apple disease detection
# x1 represents leaf spot/lesion pattern (higher value = more distinct/regular pattern)
# x2 represents leaf surface texture (higher value = smoother/healthier texture)

# Static data for healthy apples (more distinct patterns, smoother textures)
healthy_features = np.array([
    [8.5, 9.0],  # Sample 1: Clear pattern, very smooth texture
    [9.2, 8.1],  # Sample 2: Very distinct pattern, smooth texture
    [8.7, 8.8],  # Sample 3: Distinct pattern, very smooth texture
    [9.5, 8.9],  # Sample 4: Very distinct pattern, very smooth texture
    [7.8, 9.2],  # Sample 5: Distinct pattern, extremely smooth texture
    [8.1, 8.2],  # Sample 6: Distinct pattern, smooth texture
    [9.3, 8.3],  # Sample 7: Very distinct pattern, smooth texture
    [8.9, 8.8],  # Sample 8: Very distinct pattern, very smooth texture
    [8.7, 9.1],  # Sample 9: Distinct pattern, extremely smooth texture
    [8.3, 8.4],  # Sample 10: Distinct pattern, smooth texture
])
healthy_labels = np.ones(10)  # Label 1 for healthy apples

# Static data for diseased apples (irregular patterns, rough textures)
diseased_features = np.array([
    [4.2, 3.3],  # Sample 11: Irregular pattern, very rough texture
    [3.7, 4.2],  # Sample 12: Very irregular pattern, rough texture
    [4.5, 3.8],  # Sample 13: Irregular pattern, rough texture
    [3.9, 4.5],  # Sample 14: Very irregular pattern, moderately rough texture
    [4.1, 3.0],  # Sample 15: Irregular pattern, extremely rough texture
    [3.8, 3.2],  # Sample 16: Very irregular pattern, very rough texture
    [4.5, 3.5],  # Sample 17: Irregular pattern, very rough texture
    [3.3, 4.1],  # Sample 18: Extremely irregular pattern, rough texture
    [4.0, 3.8],  # Sample 19: Irregular pattern, rough texture
    [3.8, 3.6],  # Sample 20: Very irregular pattern, very rough texture
])
diseased_labels = np.ones(10) * -1  # Label -1 for diseased apples

# Combine both datasets
X = np.vstack((healthy_features, diseased_features))
y = np.hstack((healthy_labels, diseased_labels))

def create_iteration_table_image(svm_model):
    """
    Create an image of a table showing the first 10 and last 10 iterations of SVM training
    with loss values instead of error counts
    
    Parameters:
    -----------
    svm_model : SVM
        The trained SVM model with iteration history
        
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
    
    # Save the table as an image using absolute path
    image_path = os.path.join(DATA_DIR, 'iteration_history_table.png')
    plt.savefig(image_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nIteration history table image saved to: {image_path}")
    return image_path

# Print the generated dataset
print("Apple Disease Detection Dataset (Static Data):")
print("--------------------------------------------")
for i in range(len(X)):
    status = "Healthy" if y[i] == 1 else "Diseased"
    print(f"Sample {i+1}: Lesion Pattern = {X[i, 0]:.2f}, Surface Texture = {X[i, 1]:.2f}, Status: {status}")

# Visualize the data before finding hyperplane
def visualize_data_before_hyperplane():
    """
    Visualize the data before any hyperplane or decision boundary is drawn
    Shows the clear separation between the two classes and how SVM will work with this data
    """
    plt.figure(figsize=(10, 8))
    
    # Plot each class with different color and marker
    plt.scatter(healthy_features[:, 0], healthy_features[:, 1], 
                color='green', marker='o', label='Healthy Apple', 
                s=120, alpha=0.8, edgecolors='darkgreen')
    
    plt.scatter(diseased_features[:, 0], diseased_features[:, 1], 
                color='red', marker='x', label='Diseased Apple', 
                s=120, alpha=0.8, linewidth=2)
    
    plt.xlabel('Lesion Pattern (Higher = More Regular/Distinct)', fontsize=14)
    plt.ylabel('Surface Texture (Higher = Smoother)', fontsize=14)
    plt.title('Apple Disease Detection - Data Visualization Before SVM Hyperplane', fontsize=16)
    plt.legend(loc='upper right', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Save the visualization using absolute paths
    data_path_png = os.path.join(DATA_DIR, 'apple_disease_data_before_hyperplane.png')
    data_path_pdf = os.path.join(DATA_DIR, 'apple_disease_data_before_hyperplane.pdf')
    
    plt.savefig(data_path_png, dpi=300)
    plt.savefig(data_path_pdf)  # Also save as PDF for high quality
    plt.close()
    
    print("\nData visualization before hyperplane has been saved to:")
    print(f"1. {data_path_png}")
    print(f"2. {data_path_pdf}")

visualize_data_before_hyperplane()

# Train SVM model with adjusted parameters and higher C for better margin enforcement
svm = SVM(learning_rate=0.001, lambda_param=0.001, n_iterations=3000, epsilon=1e-3, C=10.0)
svm.fit(X, y)

# Create and save the iteration history as an image
create_iteration_table_image(svm)

# Make predictions
y_pred = np.array([svm.predict(x) for x in X])

# Calculate accuracy
accuracy = accuracy_score(y, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

# Create a figure with two subplots
plt.figure(figsize=(16, 8))

# First subplot: Decision Boundary and Hyperplanes
plt.subplot(1, 2, 1)

# Plot training points
plt.scatter(healthy_features[:, 0], healthy_features[:, 1], color='green', marker='o', 
            label='Healthy Apple', s=100, alpha=0.7)
plt.scatter(diseased_features[:, 0], diseased_features[:, 1], color='red', marker='x', 
            label='Diseased Apple', s=100, alpha=0.7)

# Plot decision boundary
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
Z = np.array([svm.predict(np.array([x, y])) for x, y in zip(xx.ravel(), yy.ravel())])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.2, cmap=plt.cm.RdYlGn)

# Calculate and display margin
w_norm = np.linalg.norm(svm.w)
if w_norm > 0:
    # Plot the decision boundary: w·x - b = 0
    xx_margin = np.linspace(x_min, x_max, 100)
    yy_db = (-svm.w[0] * xx_margin + svm.b) / svm.w[1]
    plt.plot(xx_margin, yy_db, 'k-', label='Decision Boundary', linewidth=2)
    
    # Get and display support vectors
    if svm.support_vectors is not None and len(svm.support_vectors) > 0:
        plt.scatter(svm.support_vectors[:, 0], svm.support_vectors[:, 1], 
                    s=200, facecolors='none', edgecolors='blue', linewidth=2,
                    label='Support Vectors')
        
        # Separate positive and negative margin lines for correct legend display
        # Positive margin (for class +1)
        yy_pos_margin = (-svm.w[0] * xx_margin + svm.b + 1) / svm.w[1]
        plt.plot(xx_margin, yy_pos_margin, 'k--', linewidth=1.5, label='Positive Margin')
        
        # Negative margin (for class -1)
        yy_neg_margin = (-svm.w[0] * xx_margin + svm.b - 1) / svm.w[1]
        plt.plot(xx_margin, yy_neg_margin, 'k--', linewidth=1.5, label='Negative Margin')
    
    # Print information about the margin
    margin_distance = svm.get_margin_distance()
    print(f"\nMargin distance: {margin_distance:.4f}")
    print(f"Number of support vectors: {len(svm.support_vectors)}")
    
    # Add margin distance text to the plot
    plt.annotate(f"Margin width: {margin_distance:.4f}", 
                 xy=(0.05, 0.05), xycoords='axes fraction', fontsize=10)
                 
    # Check for margin violations
    violations_pos = 0
    violations_neg = 0
    
    for i, x_i in enumerate(X):
        functional_margin = y[i] * (np.dot(svm.w, x_i) - svm.b)
        if y[i] == 1 and functional_margin < 1:
            violations_pos += 1
        elif y[i] == -1 and functional_margin < 1:
            violations_neg += 1
    
    print(f"Margin violations - Positive class: {violations_pos}, Negative class: {violations_neg}")
    
    # Add violation count to the plot
    plt.annotate(f"Margin violations: {violations_pos + violations_neg}", 
                 xy=(0.05, 0.10), xycoords='axes fraction', fontsize=10)

plt.xlabel('Lesion Pattern (Higher = More Regular/Distinct)', fontsize=12)
plt.ylabel('Surface Texture (Higher = Smoother)', fontsize=12)
plt.title('Apple Disease Detection - SVM Decision Boundary (Static Data)', fontsize=14)
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)

# Second subplot: Loss per epoch
plt.subplot(1, 2, 2)
plt.plot(range(1, len(svm.losses) + 1), svm.losses, 'b-', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Loss per Epoch during SVM Training', fontsize=14)
plt.grid(True, alpha=0.3)

# Add model parameters as text on loss plot
param_text = (f'Learning Rate: {svm.learning_rate}\n'
              f'Lambda: {svm.lambda_param}\n'
              f'C: {svm.C}\n'
              f'Iterations: {svm.n_iterations}')
plt.annotate(param_text, xy=(0.02, 0.95), xycoords='axes fraction', 
             fontsize=10, va='top', bbox=dict(boxstyle='round', fc='white', alpha=0.7))

plt.tight_layout()

# Save results with absolute paths
results_path_png = os.path.join(DATA_DIR, 'apple_disease_svm_static_results.png')
results_path_pdf = os.path.join(DATA_DIR, 'apple_disease_svm_static_results.pdf')
plt.savefig(results_path_png, dpi=300)
plt.savefig(results_path_pdf)

# Test with new static test samples
test_samples = [
    [8.8, 8.7],  # Should be classified as Healthy: Distinct pattern, very smooth texture
    [4.5, 3.9]   # Should be classified as Diseased: Irregular pattern, rough texture
]

print("\nTesting with static test samples:")
for i, sample in enumerate(test_samples):
    prediction = svm.predict(np.array(sample))
    status = "Healthy" if prediction == 1 else "Diseased"
    print(f"Test Sample {i+1}: Lesion Pattern = {sample[0]}, Surface Texture = {sample[1]}")
    print(f"Prediction: {status} Apple")

# Save test samples on the plot
plt.subplot(1, 2, 1)  # Back to the first subplot
for i, sample in enumerate(test_samples):
    label = f"Test Sample {i+1}"
    marker = '*' 
    color = 'cyan' if i == 0 else 'magenta'  # Different colors for each test sample
    plt.scatter(sample[0], sample[1], color=color, marker=marker, s=200, label=label)

plt.legend(loc='upper right')  # Update legend with test samples
test_results_path = os.path.join(DATA_DIR, 'apple_disease_svm_static_with_test.png')
plt.savefig(test_results_path, dpi=300)
plt.close()  # Close the plot to release memory

# Even though we can't show the plot interactively, it's saved to the files
print("\nVisualizations have been saved to:")
print(f"1. {results_path_png}")
print(f"2. {results_path_pdf}")
print(f"3. {test_results_path}")