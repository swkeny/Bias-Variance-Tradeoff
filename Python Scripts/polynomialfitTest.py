import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import matplotlib.pyplot as plt

ymin = -math.pi
ymax = math.pi
polynomialDegrees = [1, 5, 9, 14, 19]
fileRoot = "C:/Users/deifen/Documents/Projects/Bias and overfitting trade offs/Project/SampleData/sampledata "

def training_MSE_ByComplexity(dataSet, solutionSet):
      solutionBias = {}
      for i in range(1, 10):
            x = dataSet[i][1]
            y = dataSet[i][2]
            solutionBias[i] = {}
            for s in range(1, polynomialDegrees.__len__() + 1):
                  solutionBias[i][s] = np.mean((np.polyval(solutionSet[i][s], x) - y) ** 2)
      return solutionBias

def solutionBias_ByComplexity(dataSet, solutionSet,):
      solutionBias = {}
      for i in range(1, 10):
            x = dataSet[i][1]
            y = dataSet[i][2]
            solutionBias[i] = {}
            for s in range(1, polynomialDegrees.__len__() + 1):
                  solutionBias[i][s] = np.mean(np.polyval(solutionSet[i][s], x) - y)
      return solutionBias

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

xpoints = np.linspace(-math.pi, math.pi, 500)
axes = plt.gca()
axes.set_ylim([ymin,ymax])

index = 9
x= dataSets[index][1]
y = dataSets[index][2]

#******Plot Scatter Plot Of Training Data Set******
# plt.plot(x,y,'bx')

#******Plot By Polynomial Complexity For One Training Set******
# for s in range(0, polynomialDegrees.__len__()):
#       plt.plot(xpoints,solutions[index][s+1](xpoints),'r-')
#       print("Mean squared error: %.6f"
#             % np.mean((np.polyval(solutions[index][s+1], x) - y) ** 2))
#
      # print(solutions[index][s+1])
# plt.show()

#******Plot By Polynomial Complexity Across All Training Sets******
# for x in range(1, 10):
#       print('')
#       plt.plot(xpoints,solutions[x][solutionIndex](xpoints),'r-')
#       plt.plot(xpoints, sin(xpoints), 'b-')
#       print("Mean squared error: %.2f"
#             % np.mean((np.polyval(solutions[index][s+1], x) - y) ** 2))
#       print(solutions[index][s+1])
# plt.show()


# sB = training_MSE_ByComplexity(dataSets, solutions)
# for sb in range(1, polynomialDegrees.__len__()+1) :
#       print(sB[index][sb])

# sB = solutionBias_ByComplexity(dataSets, solutions)
# for bias in range(1, polynomialDegrees.__len__()+1) :
#       print("%.32f"
#             %sB[index][bias])


x = dataSets[index][1]
y = dataSets[index][2]
poly = 1
vals = []
for e in range(0, 20):
      val = np.polyval(solutions[index][poly], x[e])-y[e]
      print('%0.32f'
            %val)
      vals.append(val)

print("*************************")
print(sum(vals))
print("*************************")
print('%0.32f'
      %np.mean(np.polyval(solutions[index][poly], x) - y) )





