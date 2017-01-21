from numpy import genfromtxt
from sklearn import datasets, linear_model
import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures


fileName = "C:/Users/deifen/Documents/Projects/Bias and overfitting trade offs/Project/SampleData/sampledata 1"
fileName2 = "C:/Users/deifen/Documents/Projects/Bias and overfitting trade offs/Project/SampleData/sampledata 2"
fileName3 = "C:/Users/deifen/Documents/Projects/Bias and overfitting trade offs/Project/SampleData/sampledata 3"

df=pd.read_csv(fileName, sep=',')
df2=pd.read_csv(fileName2, sep=',')
df3=pd.read_csv(fileName3, sep=',')

vals=df.values
vals2=df2.values
vals3=df3.values

# plt.scatter(vals[:,0], vals[:,1])
# plt.show()
#
# plt.scatter(vals2[:,0], vals2[:,1])
# plt.show()
#
# plt.scatter(vals3[:,0], vals3[:,1])
# plt.show()

x1 = np.array(vals[:,0])[:, np.newaxis]
y1 = np.array(vals[:,1])

lr = linear_model.LinearRegression()
pr = linear_model.LinearRegression()
quadratic = PolynomialFeatures(degree=4)
x_quad = quadratic.fit_transform(x1)

lr.fit(x1, y1)
x_fit = np.linspace(-math.pi,math.pi, 20) [:, np.newaxis]
y_lin_fit = lr.predict(x_fit)

pr.fit(x_quad, y1)
y_quad_fit = pr.predict(quadratic.fit_transform(x_fit))




# http://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html
# The coefficients
print('Coefficients: \n', lr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % np.mean((pr.predict(quadratic.fit_transform(x_fit)) - y1) ** 2))
print(pr.coef_)
# Plot outputs
plt.scatter(x1, y1,  color='black')
plt.plot(x1, lr.predict(x1), color='blue',
          linewidth=3)
plt.plot(x_fit, y_quad_fit, label='quadratic fit', linestyle='--')
plt.xticks(())
plt.yticks(())
plt.show()
