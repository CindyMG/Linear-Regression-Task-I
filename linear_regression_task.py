# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Compute Mean Squared Error
def mean_squared_error(y_true, y_pred):
    """
    Computes the Mean Squared Error (MSE) between actual and predicted values.

    Parameters:
    - y_true: True target values (actual office prices)
    - y_pred: Predicted target values

    Returns:
    - Mean Squared Error
    """
    return np.mean((y_true - y_pred) ** 2)


# Gradient Descent algorithm
def gradient_descent(x, y, m_init, c_init, learning_rate=0.01, epochs=10):
    """
    Uses gradient descent to adjust the slope (m) and intercept (c) to fit a line.

    Parameters:
    - x: Feature values (normalized office sizes)
    - y: Target values (normalized office prices)
    - m_init, c_init: Initial values for slope and intercept
    - learning_rate: Step size for each iteration of gradient descent
    - epochs: Number of iterations to run the training

    Returns:
    - m: Final optimized slope value
    - c: Final optimized intercept value
    """
    m, c = m_init, c_init
    n = len(x)

    for epoch in range(epochs):
        y_pred = m * x + c  # Prediction using the current line equation
        m_grad = (-2 / n) * np.sum(x * (y - y_pred))  # Gradient for slope
        c_grad = (-2 / n) * np.sum(y - y_pred)  # Gradient for intercept

        # Update slope and intercept based on gradients
        m -= learning_rate * m_grad
        c -= learning_rate * c_grad

        # Calculate and display the Mean Squared Error for this iteration
        error = mean_squared_error(y, y_pred)
        print(f"Epoch {epoch + 1}: m={m:.4f}, c={c:.4f}, MSE={error:.4f}")

    return m, c


# Load the dataset
data = pd.read_csv('nairobi_office_prices.csv')  # Ensure this CSV file is available

# Extract the feature (office size) and target (office price) columns
x = data['SIZE'].values  # Office sizes (feature)
y = data['PRICE'].values  # Office prices (target)

# Normalize the feature and target values
x_mean, x_std = np.mean(x), np.std(x)
y_mean, y_std = np.mean(y), np.std(y)

x = (x - x_mean) / x_std
y = (y - y_mean) / y_std

# Set initial random values for slope (m) and intercept (c)
m_init = np.random.rand()
c_init = np.random.rand()

# Train the linear regression model for 10 epochs
m, c = gradient_descent(x, y, m_init, c_init, learning_rate=0.01, epochs=10)

# Rescale the predictions back to the original data scale
x_original = (x * x_std) + x_mean
y_pred_scaled = m * x + c
y_pred_original = (y_pred_scaled * y_std) + y_mean

# Plot the actual data points and the learned line of best fit
plt.scatter(x_original, (y * y_std) + y_mean, color='blue', label='Data points')
plt.plot(x_original, y_pred_original, color='red', label='Line of Best Fit')
plt.xlabel('Office Size (sq. ft.)')
plt.ylabel('Office Price')
plt.legend()
plt.title('Linear Regression Line of Best Fit')
plt.show()

# Predict the price for a 100 sq. ft. office
size = 100
size_scaled = (size - x_mean) / x_std
predicted_price_scaled = m * size_scaled + c
predicted_price_original = (predicted_price_scaled * y_std) + y_mean
print(f"The predicted office price for a 100 sq. ft. office is: {predicted_price_original:.2f}")
