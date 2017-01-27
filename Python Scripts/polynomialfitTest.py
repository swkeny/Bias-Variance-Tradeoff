import math
import random
from sklearn import cross_validation, linear_model
from pylab import *
import random
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

xmin = -math.pi
xmax = math.pi
ymin = -2
ymax = 2
polynomialDegrees = [1, 2, 3, 4, 13]
dataPointsPerTrainingSet = 80
#Needs numberOfTrainingSets elements in the seedMap
seedMap = [123, 232, 13, 100, 344, 45, 71, 99, 199, 80]
numberOfTrainingSets = seedMap.__len__()
# generate noisy data from an underlying function
def createSineData(n, seed):
    random.seed(seed) # for reproducibility
    x = pd.Series([random.uniform(-math.pi, math.pi) for i in range(n)])
    f = np.vectorize(lambda x: math.sin(x)) # our true function
    y = x.apply(f) # generate our labels/outputs
    e = pd.Series([random.gauss(0,1.0/3.0) for i in range(n)]) # add some noise
    y = y + e
    return x, y

def training_MSE_ByComplexity(dataSet, solutionSet):
      solutionBias = {}
      for i in range(numberOfTrainingSets):
            x = dataSet[i]['x']
            y = dataSet[i]['y']
            solutionBias[i] = {}
            for s in range(polynomialDegrees.__len__()):
                  solutionBias[i][s] = np.mean((np.polyval(solutionSet[i][s], x) - y) ** 2)
      return solutionBias

def solutionBias_ByComplexity(dataSet, solutionSet):
      avgEstimators = {}
      # for s in range(1, polynomialDegrees.__len__()+1):
      sumEst = 0
      for i in range(polynomialDegrees.__len__()):
            sumEst = sumEst + solutionSet[i][2]
      avgEstimators[s] = sumEst/numberOfTrainingSets
      return avgEstimators

dataSets = {}
for i in range(numberOfTrainingSets):
      dataSets[i] = {}
      x, y = createSineData(dataPointsPerTrainingSet, seedMap[i])
      dataSets[i]['x'] = x
      dataSets[i]['y'] = y

#Train solutions for each training data set across several degrees of polynomial complexity
solutions = {}
for x in range(numberOfTrainingSets):
      solutions[x]={}
      for s in range(polynomialDegrees.__len__()):
            x1 = dataSets[x]['x']
            y1 = dataSets[x]['y']
            func = np.polyfit(x1, y1, polynomialDegrees[s])
            solutions[x][s] = poly1d(func)

xpoints = np.linspace(-math.pi, math.pi, 500)
axes = plt.gca()
axes.set_xlim([xmin,xmax])
axes.set_ylim([ymin,ymax])

index = 1
solutionIndex = 0
x = dataSets[index]['x']
y = dataSets[index]['y']


# ******Plot Scatter Plot Of Training Data Set******
# plt.plot(x,y,'bx')

#******Plot By Polynomial Complexity For One Training Set******
# for s in range(polynomialDegrees.__len__()):
#       plt.plot(xpoints,solutions[index][s](xpoints),'r-')
#       print("Mean squared error: %.6f"
#             % np.mean((np.polyval(solutions[index][s], x) - y) ** 2))
#
#       print(solutions[index][s])
# plt.show()

# ******Plot By Polynomial Complexity Across All Training Sets******
# for i in range(numberOfTrainingSets):
#       print('')
#       plt.plot(xpoints,solutions[i][solutionIndex](xpoints),'r-')
#       plt.plot(xpoints, sin(xpoints), 'b-')
#       print("Mean squared error: %.2f"
#             % np.mean((np.polyval(solutions[index][s], x) - y) ** 2))
#       print(solutions[index][s])
# plt.show()

# ******Training MSE by complexity for a training data set******
# sB = training_MSE_ByComplexity(dataSets, solutions)
# for s in range(polynomialDegrees.__len__()) :
#       print(sB[index][s])


# !!!!!!PENDING SPLIT OF DATA INTO TEST AND TRAIN!!!!!!
# sB = solutionBias_ByComplexity(dataSets, solutions)
# for bias in range(1, polynomialDegrees.__len__()+1) :
#       print("%.32f"
#             %sB[index][bias])





