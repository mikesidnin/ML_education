import pandas as pd
import numpy as np
from sklearn import linear_model

# Read the data from csv-file
dataset = pd.read_csv("../datasets/FuelConsumptionCo2.csv")
data_frame = dataset[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY', 'FUELCONSUMPTION_COMB',
                      'CO2EMISSIONS']]

# Create test and train datasets using mask
mask = np.random.rand(len(dataset)) < 0.8
train_dataset = data_frame[mask]
test_dataset = data_frame[~mask]

# Train the train dataset
multiple_regression = linear_model.LinearRegression()
train_x = np.asanyarray(train_dataset[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY']])
train_y = np.asanyarray(train_dataset[['CO2EMISSIONS']])
multiple_regression.fit(train_x, train_y)
print('Coefficients: ', multiple_regression.coef_)

# Predict test dataset
prediction_y = multiple_regression.predict(test_dataset[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY',
                                                         'FUELCONSUMPTION_HWY']])

test_x = np.asanyarray(test_dataset[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY']])
test_y = np.asanyarray(test_dataset[['CO2EMISSIONS']])

# Get the metrics
# 1. Residual sum of squares
# 2. Explained variance score: 1 is perfect prediction
residual_sum_of_squares = np.mean((prediction_y - test_y) ** 2)
variance_score = multiple_regression.score(test_x, test_y)

print("Residual sum of squares:" + str(residual_sum_of_squares))
print("Variance score:" + str(variance_score))

# Other output implementation
# print("Residual sum of squares: %.2f" % np.mean((prediction_y - test_y) ** 2))
# print('Variance score: %.2f' % multiple_regression.score(test_x, test_y))

# TODO: Deal with "UserWarning: X has feature names, but LinearRegression was fitted without feature names"

