# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 09:02:07 2019

@author: Hrishi_Bodkhe
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_excel("Salary_Data.xlsx")

X = dataset.iloc[:, 0].values
y = dataset.iloc[:, 1].values
X = X.reshape(-1,1)
y = y.reshape(-1,1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Algorithm
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

y_predict = regressor.predict(X_test)

plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.title('EXP vs SAL')
plt.show()

plt.scatter(X_test, y_test, color='red')
plt.plot(X_test, y_predict, color='blue')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.title('EXP vs SAL')
plt.show()

y_fpre = regressor.predict(np.array([12,20,11,13]).reshape(-1,1))

#formula prediction
coef = regressor.coef_
intercept = regressor.intercept_
y_pre = intercept + (coef * 20)