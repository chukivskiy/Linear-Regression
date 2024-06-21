# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

# Get dataset
df_start = pd.read_csv('C:/Files/Documents/Дослідження операцій/Datasets/second/Life Expectancy Data.csv')
# df_start = pd.read_csv('C:/Users/Ivan/Desktop/50_Startups.csv')
print(df_start.head())

# Describe data
# print(df_start.describe())

# Data distribution
# plt.title('Profit Distribution Plot')
# sns.distplot(df_start['Profit'])
# plt.show()


# Relationship between Profit and R&D Spend
# plt.scatter(df_start['R&D Spend'], df_start['Profit'], color = 'lightcoral')
# plt.title('Profit vs R&D Spend')
# plt.xlabel('R&D Spend')
# plt.ylabel('Profit')
# plt.box(False)
# plt.show()

# # Split dataset in dependent/independent variables
# X = df_start.iloc[:, :-1].values
# y = df_start.iloc[:, -1].values
#
# # One-hot encoding of categorical data
# ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [3])], remainder = 'passthrough')
# X = np.array(ct.fit_transform(X))
#
# # Split dataset into test/train
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
#
# # Train multiple regression model
# regressor = LinearRegression()
# regressor.fit(X_train, y_train)
#
# # Predict result
# y_pred = regressor.predict(X_test)
#
# # Compare predicted result with actual value
# np.set_printoptions(precision = 2)
# result = np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1)
# # print(result)
#
# # r_squared = regressor.score(X_test, y_test)
# # print("R^2 score:", r_squared)
#
# coefficients = regressor.coef_
# print("Coefficients:", coefficients)
#
# # Вільний член (константа)
# intercept = regressor.intercept_
# print("Intercept:", intercept)