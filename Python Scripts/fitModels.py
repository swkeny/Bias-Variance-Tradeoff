import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import matplotlib.pyplot as plt

ymin = -math.pi
ymax = math.pi
polynomialDegrees = [1, 5, 9, 14]

fileRoot = "C:/Users/deifen/Documents/Projects/Bias and overfitting trade offs/Project/SampleData/sampledata "

files = {}
for x in range(1, 10):
      files[x] = fileRoot + str(x)

dataSets = {}
for x in range(1, 10):
      dataSets[x] = {}
      dataSets[x][1] = np.array(pd.read_csv(files[x], sep=',').values[:,0])
      dataSets[x][2] = np.array(pd.read_csv(files[x], sep=',').values[:,1])

solutions = {}
for x in range(1, 10):
      solutions[x]={}
      for s in range(0, polynomialDegrees.__len__()):
            x1 = dataSets[x][1]
            y1 = dataSets[x][2]
            func = np.polyfit(x1, y1, polynomialDegrees[s])
            solutions[x][s+1] = poly1d(func)

xpoints = np.linspace(-math.pi, math.pi, 100)
axes = plt.gca()
axes.set_ylim([ymin,ymax])

index = 6
x= dataSets[index][1]
y = dataSets[index][2]
plt.plot(x,y,'bx')

for s in range(0, polynomialDegrees.__len__()):
      plt.plot(xpoints,solutions[index][s+1](xpoints),'r-')
      print("Mean squared error: %.2f"
            % np.mean((np.polyval(solutions[index][s+1], x) - y) ** 2))
      print(solutions[index][s+1])
plt.show()