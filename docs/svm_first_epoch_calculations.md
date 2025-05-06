# SVM Mathematical Calculations for the First Epoch

This document explains the step-by-step mathematical calculations for the first epoch of SVM training on the apple disease detection dataset.

## Dataset Description

The apple disease detection dataset consists of 20 samples:

- 10 healthy apple samples (label +1)
- 10 diseased apple samples (label -1)

Each sample has 2 features:

- x₁: Lesion pattern (higher value = more regular/distinct pattern)
- x₂: Surface texture (higher value = smoother texture)

![Apple Disease Data Before Hyperplane](/data/apple_disease_data_before_hyperplane.png)

## SVM Initialization

The SVM model is initialized with the following parameters:

- Learning rate (α): 0.001
- Regularization parameter (λ): 0.001
- Penalty parameter (C): 10.0
- Weight vector (w): [0, 0] (initialized as zeros)
- Bias term (b): 0 (initialized as zero)

## Mathematical Formulation

The SVM optimization problem aims to find the optimal hyperplane that maximizes the margin between the two classes while minimizing classification errors.

The objective function being minimized is:

$$J(w, b) = \lambda||w||^2 + C\sum_{i=1}^{n}\max(0, 1 - y_i(w \cdot x_i - b))$$

Where:

- The first term (λ||w||²) is the regularization term to maximize the margin
- The second term (C∑max(0, 1 - yᵢ(w·xᵢ - b))) penalizes misclassifications and margin violations

## SVM Training Process Steps

The SVM training process for each sample involves the following 8 steps:

1. **Calculate Functional Margin**: Compute how confidently and correctly a data point is classified by the current hyperplane.

   - Formula: f = y(w·x - b)
   - Where y is the true label (+1 or -1), w is the weight vector, x is the feature vector, and b is the bias.
   - Higher positive values indicate confident correct classifications.

2. **Check Margin Violation**: Determine if the point violates the margin constraint.

   - If f < 1, the point violates the margin (either misclassified or inside the margin).
   - If f ≥ 1, the point is correctly classified outside the margin.

3. **Calculate Margin Distance**: For points violating the margin, calculate how far they are from satisfying the margin constraint.

   - Formula: margin_distance = 1 - f
   - This is a measure of how severely the point violates the margin constraint.
   - For the first sample in our example, this value is 1 - 0 = 1.

4. **Calculate Update Strength**: Determine how strongly to update the model for this point.

   - Formula: update_strength = min(margin_distance × C, C)
   - This scales the margin distance by the penalty parameter C (10.0 in our example)
   - For the first sample, update_strength = min(1 × 10.0, 10.0) = 10.0
   - This is NOT the margin distance itself, but rather how strongly we penalize the violation.
   - The update strength is capped at C to prevent excessive updates for severe violations.

5. **Update Alpha (Lagrange Multiplier)**: Increase the importance of this point in the model if it violates the margin.

   - Formula: α = α + learning_rate × update_strength
   - Higher alpha values indicate support vectors (points that influence the decision boundary).

6. **Calculate Loss Contribution**: Compute how much this point contributes to the overall optimization objective.

   - For margin violations: loss = λ||w||² + C × margin_distance
   - For correct classifications outside margin: loss = λ||w||²

7. **Update Weights**: Adjust weight vector to better classify this point.

   - For margin violations:
     w = w - learning_rate × (2λw - update_strength × y × x)
   - For correct classifications outside margin:
     w = w - learning_rate × (2λw)

8. **Update Bias**: Adjust the bias term to better position the hyperplane.
   - For margin violations:
     b = b - learning_rate × update_strength × y
   - For correct classifications outside margin: no change to b

### Results After All Steps

When these 8 steps are applied to all 20 samples in the dataset, the SVM model progressively adjusts its parameters to find an optimal separating hyperplane between healthy and diseased apple samples.

**For margin violations**, the model makes significant updates to the hyperplane to better classify these points, with the magnitude of update proportional to how severely they violate the margin.

**For correctly classified points** outside the margin, the model makes only small regularization updates to maintain a large margin.

By the end of one complete epoch through all 20 samples:

- The hyperplane parameters (w and b) will have moved toward a better separation of classes.
- The points with the highest influence on the decision boundary (support vectors) will have higher alpha values.
- The overall loss will decrease as the model improves its classification accuracy and margin.

The goal of these updates is to iteratively converge to a hyperplane that:

1. Correctly separates the two classes (healthy vs. diseased apples)
2. Maximizes the margin between the closest points of the two classes
3. Minimizes the classification errors, weighted by the penalty parameter C

## First Epoch Calculations

Below are the calculations for the first data point of each iteration through all 20 samples:

### Sample 1: Healthy Apple [8.5, 7.0], y = +1

1. Calculate functional margin:

   $$f = y(w \cdot x - b)$$
   $$f = +1 \times ([0, 0] \cdot [8.5, 7.0] - 0)$$
   $$f = +1 \times (0 - 0) = 0$$

2. Check margin violation:

   - Since f = 0 < 1, this point violates the margin

3. Calculate margin distance:

   $$\text{margin\_distance} = 1 - f = 1 - 0 = 1$$

4. Calculate update strength:

   $$\text{update\_strength} = \min(\text{margin\_distance} \times C, C)$$
   $$\text{update\_strength} = \min(1 \times 10.0, 10.0) = 10.0$$

5. Update alpha:

   $$\alpha_1 = \alpha_1 + \text{learning\_rate} \times \text{update\_strength}$$
   $$\alpha_1 = 0 + 0.001 \times 10.0 = 0.01$$

6. Calculate loss:

   $$\text{loss} = \lambda||w||^2 + C \times \text{margin\_distance}$$
   $$\text{loss} = 0.001 \times ||[0, 0]||^2 + 10.0 \times 1$$
   $$\text{loss} = 0 + 10.0 = 10.0$$

7. Update weights:

   $$w_1 = w_1 - \text{learning\_rate} \times (2\lambda w_1 - \text{update\_strength} \times y \times x_1)$$
   $$w_1 = 0 - 0.001 \times (2 \times 0.001 \times 0 - 10.0 \times (+1) \times 8.5)$$
   $$w_1 = 0 - 0.001 \times (-85) = 0 + 0.085 = 0.085$$

   $$w_2 = w_2 - \text{learning\_rate} \times (2\lambda w_2 - \text{update\_strength} \times y \times x_2)$$
   $$w_2 = 0 - 0.001 \times (2 \times 0.001 \times 0 - 10.0 \times (+1) \times 7.0)$$
   $$w_2 = 0 - 0.001 \times (-70) = 0 + 0.07 = 0.07$$

8. Update bias:

   $$b = b - \text{learning\_rate} \times \text{update\_strength} \times y$$
   $$b = 0 - 0.001 \times 10.0 \times (+1)$$
   $$b = 0 - 0.01 = -0.01$$

After processing Sample 1, we have:

- w = [0.085, 0.07]
- b = -0.01
- Loss contribution = 10.0

### Sample 2: Healthy Apple [9.2, 8.1], y = +1

1. Calculate functional margin:

   $$f = y(w \cdot x - b)$$
   $$f = +1 \times ([0.085, 0.07] \cdot [9.2, 8.1] - (-0.01))$$
   $$f = +1 \times ((0.085 \times 9.2) + (0.07 \times 8.1) + 0.01)$$
   $$f = +1 \times (0.782 + 0.567 + 0.01) = 1.359$$

2. Check margin violation:

   - Since f = 1.359 > 1, this point is correctly classified outside the margin

3. Update weights (only regularization term):

   $$w_1 = w_1 - \text{learning\_rate} \times (2\lambda w_1)$$
   $$w_1 = 0.085 - 0.001 \times (2 \times 0.001 \times 0.085)$$
   $$w_1 = 0.085 - 0.00000017 \approx 0.085$$

   $$w_2 = w_2 - \text{learning\_rate} \times (2\lambda w_2)$$
   $$w_2 = 0.07 - 0.001 \times (2 \times 0.001 \times 0.07)$$
   $$w_2 = 0.07 - 0.00000014 \approx 0.07$$

4. Loss contribution (only regularization term):

   $$\text{loss} = \lambda||w||^2$$
   $$\text{loss} = 0.001 \times (0.085^2 + 0.07^2)$$
   $$\text{loss} = 0.001 \times (0.007225 + 0.0049)$$
   $$\text{loss} = 0.001 \times 0.012125 = 0.000012125$$

After processing Sample 2, we have:

- w = [0.085, 0.07] (essentially unchanged due to small regularization)
- b = -0.01 (unchanged)
- Loss contribution = 0.000012125

### Sample 3: Healthy Apple [8.7, 7.8], y = +1

1. Calculate functional margin:

   $$f = y(w \cdot x - b)$$
   $$f = +1 \times ([0.085, 0.07] \cdot [8.7, 7.8] - (-0.01))$$
   $$f = +1 \times ((0.085 \times 8.7) + (0.07 \times 7.8) + 0.01)$$
   $$f = +1 \times (0.7395 + 0.546 + 0.01) = 1.2955$$

2. Check margin violation:

   - Since f = 1.2955 > 1, this point is correctly classified outside the margin

3. Update weights (only regularization term, which has minimal effect):

   - w ≈ [0.085, 0.07] (essentially unchanged)
   - b = -0.01 (unchanged)

4. Loss contribution (only regularization term):
   - Loss contribution ≈ 0.000012125

### Sample 11: Diseased Apple [5.2, 4.3], y = -1

1. Calculate functional margin:

   $$f = y(w \cdot x - b)$$
   $$f = -1 \times ([0.085, 0.07] \cdot [5.2, 4.3] - (-0.01))$$
   $$f = -1 \times ((0.085 \times 5.2) + (0.07 \times 4.3) + 0.01)$$
   $$f = -1 \times (0.442 + 0.301 + 0.01) = -0.753$$

2. Check margin violation:

   - Since f = -0.753 < 1, this point violates the margin

3. Calculate margin distance:

   $$\text{margin\_distance} = 1 - f = 1 - (-0.753) = 1.753$$

4. Calculate update strength:

   $$\text{update\_strength} = \min(\text{margin\_distance} \times C, C)$$
   $$\text{update\_strength} = \min(1.753 \times 10.0, 10.0) = 10.0$$

   (capped at C)

5. Update alpha:

   $$\alpha_{11} = 0 + 0.001 \times 10.0 = 0.01$$

6. Calculate loss:

   $$\text{loss} = \lambda||w||^2 + C \times \text{margin\_distance}$$
   $$\text{loss} = 0.001 \times (0.085^2 + 0.07^2) + 10.0 \times 1.753$$
   $$\text{loss} = 0.000012125 + 17.53 = 17.53$$

7. Update weights:

   $$w_1 = 0.085 - 0.001 \times (2 \times 0.001 \times 0.085 - 10.0 \times (-1) \times 5.2)$$
   $$w_1 = 0.085 - 0.001 \times (0.00017 - (-52))$$
   $$w_1 = 0.085 - 0.001 \times (-51.99983) = 0.085 + 0.05199983 = 0.137$$

   $$w_2 = 0.07 - 0.001 \times (2 \times 0.001 \times 0.07 - 10.0 \times (-1) \times 4.3)$$
   $$w_2 = 0.07 - 0.001 \times (0.00014 - (-43))$$
   $$w_2 = 0.07 - 0.001 \times (-42.99986) = 0.07 + 0.04299986 = 0.113$$

8. Update bias:

   $$b = -0.01 - 0.001 \times 10.0 \times (-1)$$
   $$b = -0.01 + 0.01 = 0$$

After processing Sample 11, we have:

- w = [0.137, 0.113]
- b = 0
- Loss contribution = 17.53

## Continuing Throughout the Epoch

This process continues for all 20 samples in the first epoch. After each sample:

1. The functional margin is calculated to determine if the point is correctly classified outside the margin
2. For correctly classified points outside the margin, only small regularization updates are applied
3. For margin violations, substantial updates to w and b are made to push the hyperplane toward better classification

![Apple Disease SVM Static Results](/data/apple_disease_svm_static_results.png)

## Total Loss for First Epoch

The total loss for the first epoch is calculated by summing the loss contributions from all 20 samples.

![Iteration History Table](/data/iteration_history_table.png)

## Epoch Updates Summary

By the end of the first epoch, the weight vector w and bias term b have been updated to better separate the two classes:

- Starting values: w = [0, 0], b = 0
- After first epoch: w = [w₁, w₂], b = b' (with w₁, w₂, and b' being the approximated final values)

The hyperplane equation becomes:

$$w_1 x_1 + w_2 x_2 - b = 0$$

And the two margin hyperplanes are:

$$w_1 x_1 + w_2 x_2 - b = 1$$

(positive margin)

$$w_1 x_1 + w_2 x_2 - b = -1$$

(negative margin)

With each subsequent epoch, these values are refined further until convergence or until the maximum number of iterations is reached.
