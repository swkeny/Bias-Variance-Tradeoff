import math
import random
from sklearn import cross_validation, linear_model
from sklearn.cross_validation import train_test_split
from pylab import *
import random
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

xmin = -math.pi
xmax = math.pi
ymin = -2
ymax = 2
polynomialDegrees = [1, 2, 3, 5, 12]
dataPointsPerTrainingSet = 100
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
    return np.stack((x, y), axis=-1)

# Calculate Training MSE
def training_MSE_ByComplexity(dataSetTrain, solutionSet):
      avgMSE = {}
      for i in range(numberOfTrainingSets):
            x = dataSetTrain[i]['x']
            y = dataSetTrain[i]['y']
            avgMSE[i] = {}
            for s in range(polynomialDegrees.__len__()):
                  avgMSE[i][s] = np.mean((np.polyval(solutionSet[i][s], x) - y) ** 2)
      return avgMSE

def avgFittedFunction(solutionSet):
      avgEstimators = {}
      for s in range(polynomialDegrees.__len__()):
            sumEst = 0
            for i in range(numberOfTrainingSets):
                  sumEst = sumEst + solutionSet[i][s]
            avgEstimators[s] = sumEst / numberOfTrainingSets
      return avgEstimators

def solutionBias_ByComplexity(dataSetTest, solutionSet):
      solutionBias = {}
      avgEstimators = avgFittedFunction(solutionSet)
      for s in range(polynomialDegrees.__len__()):
            solutionBiasPerDataSet = []
            x = pd.Series([random.uniform(-math.pi, math.pi) for i in range(dataPointsPerTrainingSet)])
            f = np.vectorize(lambda x: math.sin(x))  # our true function
            y = x.apply(f)
            solutionBias[s] = np.mean(np.polyval(avgEstimators[s], x)-y)
      return solutionBias

dataSetsTrain = {}
dataSetsTest = {}
for i in range(numberOfTrainingSets):
      dataSetsTrain[i] = {}
      dataSetsTest[i] = {}
      ds = createSineData(dataPointsPerTrainingSet, seedMap[i])
      train, test = train_test_split(ds, test_size=0.2)
      dataSetsTrain[i]['x'] = train[:,0]
      dataSetsTrain[i]['y'] = train[:,1]
      dataSetsTest[i]['x'] = test[:,0]
      dataSetsTest[i]['y'] = test[:,1]

#Train solutions for each training data set across several degrees of polynomial complexity
solutions = {}
for x in range(numberOfTrainingSets):
      solutions[x]={}
      for s in range(polynomialDegrees.__len__()):
            x1 = dataSetsTrain[x]['x']
            y1 = dataSetsTrain[x]['y']
            func = np.polyfit(x1, y1, polynomialDegrees[s])
            solutions[x][s] = poly1d(func)

xpoints = np.linspace(-math.pi, math.pi, 500)
axes = plt.gca()
axes.set_xlim([xmin,xmax])
axes.set_ylim([ymin,ymax])

index = 0
solutionIndex = 4
x = dataSetsTrain[index]['x']
y = dataSetsTrain[index]['y']


# ******Plot Scatter Plot Of Training Data Set******
# plt.plot(x,y,'bx')

#******Plot By Polynomial Complexity For One Training Set******
# for s in range(polynomialDegrees.__len__()):
#       plt.plot(xpoints,solutions[index][s](xpoints),'r-')
#       print("Mean squared error: %.6f"
#             % np.mean((np.polyval(solutions[index][s], x) - y) ** 2))
#       print(solutions[index][s])
# plt.show()

# ******Plot By Polynomial Complexity Across All Training Sets with Average Fit******
# plt.plot(xpoints, sin(xpoints), 'k-', linewidth=4)
# plt.plot(xpoints, avgFittedFunction(solutions)[solutionIndex](xpoints), 'g-', linewidth=4)
# for i in range(numberOfTrainingSets):
#       print('')
#       plt.plot(xpoints,solutions[i][solutionIndex](xpoints),'r-')
      # print(solutions[index][s])
#       print(training_MSE_ByComplexity(dataSetsTrain, solutions)[index][s])
# plt.show()

# # ******Plot By Polynomial Complexity Across All Training Sets with Average Fit******
# plt.plot(xpoints, sin(xpoints), 'k-', linewidth=4)
# for s in range(polynomialDegrees.__len__()):
#       plt.plot(xpoints, avgFittedFunction(solutions)[s](xpoints), 'g-', linewidth=2)
# plt.show()


# ******Training MSE by complexity for a training data set******
# trainingMSE = training_MSE_ByComplexity(dataSetsTrain, solutions)
# for s in range(polynomialDegrees.__len__()) :
#       print(trainingMSE[index][s])


# Prints solution bias by complexity
# solutionBias = solutionBias_ByComplexity(dataSetsTest, solutions)
# for s in range(polynomialDegrees.__len__()) :
#       print(solutionBias[s])




#VALIDATIONS**********************
# Validate avg fitted function
# print(solutions)
# print(avgFittedFunction(solutions)[2])

