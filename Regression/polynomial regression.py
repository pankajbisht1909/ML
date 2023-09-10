import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("D:\python\Position_Salaries.csv")
dataset

X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# fitting the linear regression model
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X, y)

# visualising the linear regression model
plt.scatter(X, y, color='red')
plt.plot(X, lin_reg.predict(X), color='blue')
plt.title("Truth or Bluff(Linear)")
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# polynomial regression model
from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree=2)
X_poly = poly_reg.fit_transform(X)

X_poly  # prints X_poly

lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)

# visualising polynomial regression
from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)

X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color='red')

plt.plot(X_grid, lin_reg2.predict(poly_reg.fit_transform(X_grid)), color='blue')

plt.title("Truth or Bluff(Polynomial)")
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()