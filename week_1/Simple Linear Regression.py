import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score

dataset = pd.read_csv("FuelConsumptionCo2.csv")

dataset_columns = dataset[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]

# Step 1
# Let's split our dataset into train and test sets.
# 80% of the entire dataset will be used for training and 20% for testing.
# We create a mask to select random rows using np.random.rand() function:

msk = np.random.rand(len(dataset)) < 0.8
train = dataset_columns[msk]
test = dataset_columns[~msk]

# Step 2
# Train data distribution
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

# Step 3
# Using sklearn package to model data.
regression = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regression.fit(train_x, train_y)

theta_1 = regression.coef_
theta_0 = regression.intercept_
print('Coefficients: ', theta_1)
print('Intercept: ', theta_0)

# Step 4
# Plot the fit line over the data
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.plot(train_x, theta_1[0][0]*train_x + theta_0[0], '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

# Step 5
# Get the Evaluation metrics
test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
predicted_test_y = regression.predict(test_x)

mae_metric = np.mean(np.absolute(predicted_test_y - test_y))
mse_metric = np.mean((predicted_test_y - test_y) ** 2)
r2_score_metric = r2_score(test_y, predicted_test_y)

print("---------------CO2EMISSIONS prediction metrics by ENGINESIZE-------------------")
print("Mean absolute error: " + str(mae_metric))
print("Residual sum of squares (MSE): " + str(mse_metric))
print("R2-score: " + str(r2_score_metric))
# Short version
# print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
# print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
# print("R2-score: %.2f" % r2_score(test_y , test_y_) )

# Exercise

# Getting new train and test arrays
fuel_train_x = np.asanyarray(train[['FUELCONSUMPTION_COMB']])
fuel_test_x = np.asanyarray(test[['FUELCONSUMPTION_COMB']])

# Train by new X and old Y
fuel_regression = linear_model.LinearRegression()
fuel_regression.fit(fuel_train_x, train_y)

# Predict
fuel_predictions = fuel_regression.predict(fuel_test_x)

# Metrics
fuel_mae_metric = np.mean(np.absolute(fuel_predictions - test_y))
fuel_mse_metric = np.mean((fuel_predictions - test_y) ** 2)
fuel_r2_score_metric = r2_score(test_y, fuel_predictions)

print("---------------CO2EMISSIONS prediction metrics by FUELCONSUMPTION_COMB-------------------")
print("Mean absolute error: " + str(fuel_mae_metric))
print("Residual sum of squares (MSE): " + str(fuel_mse_metric))
print("R2-score: " + str(fuel_r2_score_metric))
