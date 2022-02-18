import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures

from week_2.functions.functions_methods import sigmoid

# Preparing dataset
dataset = pd.read_csv("../datasets/china_gdp.csv")
dataframe_x = dataset['Year'].values
dataframe_y = dataset['Value'].values

# Normalize data
dataframe_x_normalized = dataframe_x / max(dataframe_x)
dataframe_y_normalized = dataframe_y / max(dataframe_y)

# Split normalized data to test/train
mask = np.random.rand(len(dataset)) < 0.8
train_x = dataframe_x_normalized[mask]
train_y = dataframe_y_normalized[mask]
test_x = dataframe_x_normalized[~mask]
test_y = dataframe_y_normalized[~mask]
# TODO: deal with polynomial and compare it with sigmoid
# Polynomial case
polynomial_degree_feature = PolynomialFeatures(degree=3)
train_x_double_array = np.asanyarray(dataset[['Year']])
train_y_double_array = np.asanyarray(dataset[['Value']])
train_x_pol = dataframe_x_normalized[mask]
train_y_pol = dataframe_y_normalized[mask]
test_x_pol = dataframe_x_normalized[~mask]
test_y_pol = dataframe_y_normalized[~mask]
train_x_transformed = polynomial_degree_feature.fit_transform(train_x_double_array)
linear_regression_model = linear_model.LinearRegression()

# We can use curve_fit which uses non-linear least squares to fit our sigmoid function, to data.
# popt, pcov = curve_fit(sigmoid, train_x, train_y)
#
# popt - array
# Optimal values for the parameters so that the sum of the squared residuals of f(xdata, *popt) - ydata is minimized.
# pcov - 2-D array
# The estimated covariance of popt. The diagonals provide the variance of the parameter estimate.
# To compute one standard deviation errors on the parameters use perr = np.sqrt(np.diag(pcov)).
beta_optimal_sigmoid, beta_covariance_sigmoid = curve_fit(sigmoid, train_x, train_y)

fit_y_sigmoid = sigmoid(test_x, beta_optimal_sigmoid[0], beta_optimal_sigmoid[1])
fit_y_polynomial = linear_regression_model.fit(train_x_transformed, train_y_double_array)
# fit_y_exponential = exponential(test_x, beta_optimal_exponential[0], beta_optimal_exponential[1],
#                                beta_optimal_exponential[2])

plt.scatter(train_x, train_y, color='red', s=1.1)

plot_x = np.arange(min(train_x), max(train_x), 0.0001)

plot_y_sigmoid = sigmoid(plot_x, beta_optimal_sigmoid[0], beta_optimal_sigmoid[1])
plot_y_polynomial = linear_regression_model.intercept_[0] + \
         linear_regression_model.coef_[0][1]*plot_x + \
         linear_regression_model.coef_[0][2]*np.power(plot_x, 2) + \
         linear_regression_model.coef_[0][3]*np.power(plot_x, 3)

plt.plot(plot_x, plot_y_sigmoid, '-b')
plt.plot(plot_x, plot_y_polynomial, '-g')
plt.show()

print(type(fit_y_polynomial))

print("MAE sigmoid: %.5f" % np.mean(np.absolute(fit_y_sigmoid - test_y)))
print("MAE exponential: %.5f" % np.mean(np.absolute(fit_y_polynomial - test_y)))

print("MSE sigmoid: %.5f" % np.mean((fit_y_sigmoid - test_y) ** 2))
print("MSE exponential: %.5f" % np.mean((fit_y_polynomial - test_y) ** 2))

print("R2-score sigmoid: %.5f" % r2_score(test_y, fit_y_sigmoid))
print("R2-score exponential: %.5f" % r2_score(test_y, fit_y_polynomial))
