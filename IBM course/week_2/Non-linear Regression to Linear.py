import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures

# Read Dataset
dataset = pd.read_csv("../datasets/FuelConsumptionCo2.csv")
dataframe = dataset[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]

# Split to test and train data
mask = np.random.rand(len(dataframe)) < 0.8
train = dataframe[mask]
test = dataframe[~mask]

# Get the lists of x and y train/test data
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])

# PolynomialFeatures() function in Scikit-learn library, drives a new feature sets from the original feature set.
# A matrix will be generated consisting of all polynomial combinations of
# the features with degree less than or equal to the specified degree.
# fit_transform takes our x values, and output a list of our data raised from power of 0 to power of 2
polynomial_degree_feature = PolynomialFeatures(degree=3)
train_x_transformed = polynomial_degree_feature.fit_transform(train_x)

# Now we need to find coefficients by using linear model
# Non-linear stuff is hidden in x values array
# Train linear model using array of x values transformed before
linear_regression_model = linear_model.LinearRegression()
train_y_fitted = linear_regression_model.fit(train_x_transformed, train_y)

# Plot a function with fitted values and coefficients---------------------------------
# on top of the data distribution
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='red', s=1.1)

# np.arange(start, stop, step, dtype)
# # Example:
# np.arange(10,30,2)
# array([10, 12, 14, 16, 18, 20, 22, 24, 26, 28])
plot_x = np.arange(0.0, 10.0, 0.0001)

# Multiple Linear Function
plot_y = linear_regression_model.intercept_[0] + \
         linear_regression_model.coef_[0][1]*plot_x + \
         linear_regression_model.coef_[0][2]*np.power(plot_x, 2) + \
         linear_regression_model.coef_[0][3]*np.power(plot_x, 3)

plt.plot(plot_x, plot_y, '-b')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()
# ------------------------------------------------------------------------------------

# Predicting on test dataset and getting metrics
test_x_transformed = polynomial_degree_feature.transform(test_x)
test_y_fitted = linear_regression_model.predict(test_x_transformed)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_fitted - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_fitted - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y, test_y_fitted))
