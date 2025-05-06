import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iterations=1000, epsilon=1e-5, C=1.0):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iterations = n_iterations
        self.epsilon = epsilon  # Threshold to identify support vectors
        self.C = C  # Regularization parameter (higher C = less margin violations)
        self.w = None
        self.b = None
        self.X = None
        self.y = None
        self.alphas = None  # Lagrange multipliers
        self.support_vectors = None
        self.support_vector_labels = None
        self.support_vector_indices = None
        self.losses = []
        # Add history tracking for iterations
        self.iteration_history = []
        
    def fit(self, X, y):
        """
        Train the SVM model using gradient descent with better margin enforcement
        
        Parameters:
        -----------
        X : numpy.ndarray
            Training features, shape (n_samples, n_features)
        y : numpy.ndarray
            Target labels, shape (n_samples,), values should be -1 or 1
        """
        n_samples, n_features = X.shape
        
        # Store training data
        self.X = X
        self.y = np.where(y <= 0, -1, 1)  # Ensure y is in correct format (-1, 1)
        
        # Initialize weights, bias, and alphas
        self.w = np.zeros(n_features)
        self.b = 0
        self.alphas = np.zeros(n_samples)
        self.losses = []
        self.iteration_history = []
        
        # Gradient descent with stronger margin enforcement
        for i in range(self.n_iterations):
            epoch_loss = 0
            for idx, x_i in enumerate(X):
                # Calculate functional margin
                functional_margin = self.y[idx] * (np.dot(x_i, self.w) - self.b)
                
                # Check if this point violates or is on the margin
                if functional_margin >= 1:
                    # Point is correctly classified and outside the margin
                    self.alphas[idx] = 0  # Not a support vector
                    epoch_loss += self.lambda_param * np.sum(self.w ** 2)
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w)
                else:
                    # Point is either misclassified or inside the margin
                    # Stronger update for points far inside the margin or misclassified
                    margin_distance = 1 - functional_margin
                    update_strength = min(margin_distance * self.C, self.C)  # Cap the update strength
                    
                    self.alphas[idx] += self.learning_rate * update_strength  # Update alpha
                    
                    # Loss includes the margin violation cost
                    epoch_loss += self.lambda_param * np.sum(self.w ** 2) + self.C * margin_distance
                    
                    # Stronger gradient updates for margin violations
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w - update_strength * self.y[idx] * x_i)
                    self.b -= self.learning_rate * update_strength * self.y[idx]
            
            # Calculate predictions and error rates with more precision (as a decimal)
            predictions = np.array([np.sign(np.dot(x_i, self.w) - self.b) for x_i in X])
            misclassifications = np.sum(predictions != self.y)
            
            # Calculate error rate (proportion of misclassified samples)
            error_rate = misclassifications / n_samples
            
            # Also calculate margin violations (points inside or on the wrong side of the margin)
            functional_margins = self.y * (np.dot(X, self.w) - self.b)
            margin_violations = np.sum(functional_margins < 1) / n_samples
            
            # Save the average loss for this epoch
            avg_loss = epoch_loss / n_samples
            self.losses.append(avg_loss)
            
            # Save iteration history with detailed error metrics
            self.iteration_history.append({
                'epoch': i+1,
                'w': self.w.copy(),  # Create a copy to avoid reference issues
                'b': self.b,
                'misclassifications': misclassifications,
                'error_rate': error_rate,
                'margin_violations': margin_violations
            })
            
            # Optional: Early stopping if loss improvement is minimal
            if i > 0 and abs(self.losses[i] - self.losses[i-1]) < 1e-5:
                print(f"Early stopping at iteration {i+1} due to convergence.")
                break
        
        # Identify support vectors (points with non-zero Lagrange multipliers)
        self.support_vector_indices = np.where(self.alphas > self.epsilon)[0]
        
        # If not enough support vectors were found, use the points closest to the margin
        if len(self.support_vector_indices) < 2:
            # Calculate functional margins for all points
            functional_margins = self.y * (np.dot(X, self.w) - self.b)
            
            # Find points closest to the margin on both sides (positive and negative)
            positive_class = np.where(self.y == 1)[0]
            negative_class = np.where(self.y == -1)[0]
            
            # Get the closest points from each class
            if len(positive_class) > 0:
                closest_positive = positive_class[np.argmin(abs(functional_margins[positive_class] - 1))]
            else:
                closest_positive = None
                
            if len(negative_class) > 0:
                closest_negative = negative_class[np.argmin(abs(functional_margins[negative_class] - 1))]
            else:
                closest_negative = None
                
            # Combine the closest points
            closest_points = []
            if closest_positive is not None:
                closest_points.append(closest_positive)
            if closest_negative is not None:
                closest_points.append(closest_negative)
                
            # Add more points if needed
            if len(closest_points) < 2:
                closest_points = np.argsort(abs(functional_margins - 1))[:2]
                
            self.support_vector_indices = np.array(closest_points)
        
        # Extract the support vectors and their labels
        self.support_vectors = X[self.support_vector_indices]
        self.support_vector_labels = self.y[self.support_vector_indices]
        
    def get_iteration_table(self):
        """
        Get a table of the first 10 and last 10 iterations
        
        Returns:
        --------
        list: First 10 iterations
        list: Last 10 iterations
        """
        if not self.iteration_history:
            return [], []
            
        first_iterations = self.iteration_history[:10]
        
        # For last iterations, take the last 10 or all if less than 10
        if len(self.iteration_history) <= 10:
            last_iterations = []
        else:
            last_iterations = self.iteration_history[-10:]
            
        return first_iterations, last_iterations

    def predict(self, X):
        """
        Make predictions using the trained SVM model
        
        Parameters:
        -----------
        X : numpy.ndarray
            Test features, shape (n_samples, n_features) or a single sample
            
        Returns:
        --------
        numpy.ndarray or float
            Predicted labels, shape (n_samples,) or a single label
        """
        # Handle both single samples and multiple samples
        if X.ndim == 1:
            X = X.reshape(1, -1)
            
        linear_output = np.dot(X, self.w) - self.b
        return np.sign(linear_output)
    
    def get_margin_distance(self):
        """
        Calculate the margin distance (distance between the two hyperplanes)
        
        Returns:
        --------
        float
            The margin distance
        """
        w_norm = np.linalg.norm(self.w)
        if w_norm > 0:
            return 2 / w_norm
        return 0
        
    def get_positive_margin_points(self):
        """
        Get points that are closest to the positive margin
        
        Returns:
        --------
        numpy.ndarray
            Support vectors on the positive margin
        """
        if self.support_vectors is None or len(self.support_vectors) == 0:
            return None
            
        positive_sv_indices = np.where(self.support_vector_labels == 1)[0]
        if len(positive_sv_indices) == 0:
            return None
            
        return self.support_vectors[positive_sv_indices]
        
    def get_negative_margin_points(self):
        """
        Get points that are closest to the negative margin
        
        Returns:
        --------
        numpy.ndarray
            Support vectors on the negative margin
        """
        if self.support_vectors is None or len(self.support_vectors) == 0:
            return None
            
        negative_sv_indices = np.where(self.support_vector_labels == -1)[0]
        if len(negative_sv_indices) == 0:
            return None
            
        return self.support_vectors[negative_sv_indices]
