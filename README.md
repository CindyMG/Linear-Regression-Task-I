# Linear-Regression-Task-I
# Nairobi Office Price Prediction using Linear Regression

This project involves training a simple linear regression model using Gradient Descent to predict office prices based on office sizes from a dataset of Nairobi office spaces.

## Task Overview

Given a dataset with one feature (office size in square feet) and one target (office price), the goal is to:
- Write two Python functions:
  1. One for computing Mean Squared Error (MSE) to serve as a performance measure.
  2. Another for performing Gradient Descent to update weights (slope `m` and intercept `c`).
- Train the linear regression model for 10 epochs, showing the MSE after each epoch.
- Plot the line of best fit after training.
- Use the learned model to predict the office price for a 100 sq. ft. office.

## Approach

1. **Mean Squared Error (MSE)**: A Python function that calculates the average squared difference between the actual and predicted office prices.
2. **Gradient Descent**: A function that updates the slope (`m`) and intercept (`c`) of the linear model over 10 epochs by minimizing the error between the actual and predicted prices.
3. **Feature Scaling**: Both the feature (office size) and target (office price) are normalized to ensure better convergence during gradient descent.
4. **Rescaling**: After training, the predicted values are converted back to the original scale for proper interpretation.
5. **Prediction**: The model is used to predict the office price for a 100 sq. ft. office.

## How to Run

1. Ensure the dataset (`nairobi_office_prices.csv`) is in the same directory as the script.
2. Run the Python script `linear_regression_task.py`.
3. View the Mean Squared Error at each epoch and observe the plot of the line of best fit.
4. See the predicted price for a 100 sq. ft. office.

## Libraries Used

- `numpy`: For numerical operations.
- `pandas`: For data manipulation.
- `matplotlib`: For plotting the line of best fit.

## Example Output

After training for 10 epochs, the predicted price for a 100 sq. ft. office is displayed, and a plot shows the data points and the line of best fit.


## Conclusion

This project demonstrates how to implement a simple linear regression model from scratch using Python, with key elements like MSE calculation, Gradient Descent optimization, and feature scaling for improved training efficiency.
