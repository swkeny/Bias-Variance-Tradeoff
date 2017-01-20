from numpy import genfromtxt
from sklearn import datasets, linear_model
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


fileName = "C:/Users/deifen/Documents/Projects/Bias and overfitting trade offs/Bias and Variance Tradeoff Project/SampleData/dataSample 1"
fileName2 = "C:/Users/deifen/Documents/Projects/Bias and overfitting trade offs/Bias and Variance Tradeoff Project/SampleData/dataSample 2"
fileName3 = "C:/Users/deifen/Documents/Projects/Bias and overfitting trade offs/Bias and Variance Tradeoff Project/SampleData/dataSample 3"

#my_data = genfromtxt(fileName, delimiter=',')

df=pd.read_csv(fileName, sep=',')
df2=pd.read_csv(fileName2, sep=',')
df3=pd.read_csv(fileName3, sep=',')

vals=df.values
vals2=df2.values
vals3=df3.values

plt.scatter(vals[:,0], vals[:,1])
# plt.show()

plt.scatter(vals2[:,0], vals2[:,1])
# plt.show()

plt.scatter(vals3[:,0], vals3[:,1])
# plt.show()

x1 = np.array(vals[:,0])[:, np.newaxis]
y1 = np.array(vals[:,1])
lr = linear_model.LinearRegression()

lr.fit(x1, y1)

# http://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html
# The coefficients
print('Coefficients: \n', lr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % np.mean((lr.predict(x1) - y1) ** 2))

# Plot outputs
plt.scatter(x1, y1,  color='black')
plt.plot(x1, lr.predict(x1), color='blue',
          linewidth=3)
plt.xticks(())
plt.yticks(())
plt.show()