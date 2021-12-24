import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from week_2.functions.functions_methods import sigmoid


# Preparing dataset
dataset = pd.read_csv("../datasets/china_gdp.csv")
dataframe_x = dataset['Year'].values
dataframe_y = dataset['Value'].values


# Normalize data
dataframe_x_normalized = dataframe_x/max(dataframe_x)
dataframe_y_normalized = dataframe_y/max(dataframe_y)

# Split normalized data to test/train
mask = np.random.rand(len(dataset)) < 0.8
train_x = dataframe_x_normalized[mask]
train_y = dataframe_y_normalized[mask]
test_x = dataframe_x_normalized[~mask]
test_y = dataframe_y_normalized[~mask]


# We can use curve_fit which uses non-linear least squares to fit our sigmoid function, to data.
# popt, pcov = curve_fit(sigmoid, train_x, train_y)
#
# popt - array
# Optimal values for the parameters so that the sum of the squared residuals of f(xdata, *popt) - ydata is minimized.
# pcov - 2-D array
# The estimated covariance of popt. The diagonals provide the variance of the parameter estimate.
# To compute one standard deviation errors on the parameters use perr = np.sqrt(np.diag(pcov)).
beta_optimal, beta_covariance = curve_fit(sigmoid, train_x, train_y)

fit_y = sigmoid(test_x, beta_optimal[0], beta_optimal[1])

print("Mean absolute error: %.2f" % np.mean(np.absolute(fit_y - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((fit_y - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y, fit_y))
