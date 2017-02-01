import math
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
polynomialDegrees = [1, 2, 3, 6, 10]
dataPointsPerTrainingSet = 40
testSplit = 0.2

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
def MSE_ByComplexity(dataSet, solutionSet):
      avgMSE = {}
      for i in range(numberOfTrainingSets):
            x = dataSet[i]['x']
            y = dataSet[i]['y']
            avgMSE[i] = {}
            for s in range(polynomialDegrees.__len__()):
                  avgMSE[i][s] = np.mean((np.polyval(solutionSet[i][s], x) - y) ** 2)

      return avgMSE

def transformArray(array, len):
      transformedArray = {}
      for i in range(len):
            transformedArray[i] = {}
            for j in range(numberOfTrainingSets):
                  transformedArray[i][j] = array[j][i]
      return transformedArray
#For a given solution, average squared error across all datasets
def MSE_toPlot(xpoints, solutionSet, solutionIndex, index):
      meanSquaredError = {}
      result = {}
      len = xpoints.__len__()
      x = pd.Series(xpoints)
      f = np.vectorize(lambda x: math.sin(x))
      y = x.apply(f)
      for i in range(numberOfTrainingSets):
            yhat = solutions[i][solutionIndex](xpoints)
            meanSquaredError[i]= (yhat - y) ** 2
      transformedMeanSquareError = transformArray(meanSquaredError, len)
      for row in range(len):
            sum = 0
            for column in range(numberOfTrainingSets):
                  sum = sum + transformedMeanSquareError[row][column]
            result[row] = sum/numberOfTrainingSets
      return result

def avgFittedFunction(solutionSet):
      avgEstimators = {}
      for s in range(polynomialDegrees.__len__()):
            sumEst = 0
            for i in range(numberOfTrainingSets):
                  sumEst = sumEst + solutionSet[i][s]
            avgEstimators[s] = sumEst / numberOfTrainingSets
      return avgEstimators
#Not sure if this one is riight......
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
      train, test = train_test_split(ds, test_size=testSplit)
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



xpoints = np.linspace(-math.pi, math.pi, 100)
axes = plt.gca()
axes.set_xlim([xmin,xmax])
axes.set_ylim([ymin,ymax])

index = 0
solutionIndex = 4
x = dataSetsTrain[index]['x']
y = dataSetsTrain[index]['y']


# ******Plot Scatter Plot Of Training Data Set******
# plt.plot(x,y,'bx')

# ******REMOVE Plot By Polynomial Complexity For One Training Set******
# for s in range(polynomialDegrees.__len__()):
#       plt.plot(xpoints,solutions[index][s](xpoints),'r-')
#       print("Mean squared error: %.6f"
#             % np.mean((np.polyval(solutions[index][s], x) - y) ** 2))
#       print(solutions[index][s])
# plt.show()
#
#
# ax1 = plt.subplot(131)
# ax1.plot([1,2], [3,4])
# ax1.set_xlim([0, 5])
# ax1.set_ylim([0, 5])
#
#
# ax2 = plt.subplot(132)
# ax2.scatter([1, 2],[3, 4])
# ax2.set_xlim([0, 5])
# ax2.set_ylim([0, 5])
# plt.show()

# ******Plot By Polynomial Complexity Across All Training Sets with Average Fit******
# error = MSE_toPlot(xpoints, solutions, solutionIndex, index)
# #print(pd.Series(error))
# ax1 = plt.subplot(311)
# ax1.plot(xpoints, sin(xpoints), 'k-', linewidth=4)
# ax1.plot(xpoints, avgFittedFunction(solutions)[solutionIndex](xpoints), 'g-', linewidth=4)
# ax1.set_xlim([xmin,xmax])
# ax1.set_ylim([-1.25,1.25])
# for i in range(numberOfTrainingSets):
#       ax1.plot(xpoints,solutions[i][solutionIndex](xpoints),'y-')
# ax2 = plt.subplot(312, sharex=ax1)
# plt.plot(xpoints, pd.Series(error), 'r--', linewidth=4)
# ax2.set_xlim([xmin,xmax])
# ax2.set_ylim([0,.5])
# plt.show()

# ******Plot By Polynomial Complexity Across All Training Sets with Average Fit******
# plt.plot(xpoints, sin(xpoints), 'k-', linewidth=4)
#
# for s in range(len(polynomialDegrees)):
#       plt.plot(xpoints, avgFittedFunction(solutions)[s](xpoints), 'g-', linewidth=2)
# plt.show()


# ******Training MSE by complexity for a training data set******
# trainingMSE = MSE_ByComplexity(dataSetsTest, solutions)
# for s in range(len(polynomialDegrees)) :
#       print(trainingMSE[index][s])




# NOT SURE IF THIS IS RIGHT! Prints solution bias by complexity
# solutionBias = solutionBias_ByComplexity(dataSetsTest, solutions)
# for s in range(polynomialDegrees.__len__()) :
#       print(solutionBias[s])




#Bias and variance by complexity
#Setup
polynomialDegrees = np.linspace(1, 12, endpoint=True, num=12)
# polynomialDegrees = [12]
dataPointsPerTrainingSet = 60
testSplit = .1
xpoints = np.linspace(-math.pi, math.pi, 200)
index=0
seedMap = np.linspace(1, 100,  endpoint=True, num=100)
numberOfTrainingSets = seedMap.__len__()

dSTrain = {}
dSTest = {}
for i in range(numberOfTrainingSets):
      dSTrain[i] = {}
      dSTest[i] = {}
      dataSet = createSineData(dataPointsPerTrainingSet, seedMap[i])
      train, test = train_test_split(dataSet, test_size=testSplit)
      dSTrain[i]['x'] = train[:,0]
      dSTrain[i]['y'] = train[:,1]
      dSTest[i]['x'] = test[:,0]
      dSTest[i]['y'] = test[:,1]

solutionSet = {}
for i in range(numberOfTrainingSets):
      solutionSet[i]={}
      for s in range(polynomialDegrees.__len__()):
            x2 = dSTrain[i]['x']
            y2 = dSTrain[i]['y']
            f1 = np.polyfit(x2, y2, polynomialDegrees[s])
            solutionSet[i][s] = poly1d(f1)


# Training Error
# trainingError = {}
# for s in range(polynomialDegrees.__len__()):
#       avg = 0
#       for i in range(numberOfTrainingSets):
#             avg = avg + np.mean((np.polyval(solutionSet[i][s], dSTrain[i]['x']) - dSTrain[i]['y']) ** 2)
#       trainingError[s+1] = avg/numberOfTrainingSets
#
# axes = plt.gca()
# axes.set_xlim([0,polynomialDegrees.__len__()+1])
# axes.set_ylim([0,np.max(list(trainingError.values()))+.25])
#
# plt.plot(list(trainingError.keys()), list(trainingError.values()), 'b-')
# plt.show()



# Test Error
# testError = {}
# for s in range(polynomialDegrees.__len__()):
#       avg = 0
#       for i in range(numberOfTrainingSets):
#             avg = avg + np.mean((np.polyval(solutionSet[i][s], dSTest[i]['x']) - dSTest[i]['y']) ** 2)
#             testError[s+1] = avg/numberOfTrainingSets
#
# axes = plt.gca()
# axes.set_xlim([0,polynomialDegrees.__len__()+1])
# axes.set_ylim([0,np.max(list(testError.values()))+.25])
# plt.plot(list(testError.keys()), list(testError.values()), 'r--')
# plt.show()

#Bias Squared
# biasSquared = {}
# x = np.linspace(-math.pi, math.pi, 100)
# f = np.vectorize(lambda x: math.sin(x))
# y = x.apply(f)
# for s in range(polynomialDegrees.__len__()):
#       sum = 0
#       for i in range(numberOfTrainingSets):
#             sum = sum + np.mean((np.polyval(solutionSet[i][s], dSTest[i]['x']) - y) ** 2)
#             biasSquared[s+1] = sum/numberOfTrainingSets
#
# axes = plt.gca()
# axes.set_xlim([0,polynomialDegrees.__len__()+1])
# axes.set_ylim([0,np.max(list(biasSquared.values()))+.25])
#
# plt.plot(list(testError.keys()), list(testError.values()), 'b--')
# plt.plot(list(biasSquared.keys()), list(biasSquared.values()), 'r--')
# plt.show()


#Combined avg fit, true function, and scatterplots
solutionIndex=5


for i in range(numberOfTrainingSets):
      plt.scatter(dSTest[i]['x'], dSTest[i]['y'], color='r')
      plt.scatter(dSTest[i]['x'], np.polyval(solutionSet[i][solutionIndex], dSTest[i]['x']), color='y')
plt.plot(xpoints, sin(xpoints), 'b--')
plt.plot(xpoints, np.polyval(avgFittedFunction(solutionSet)[solutionIndex], xpoints), 'g-')
plt.show()


#VALIDATIONS**********************
# Validate avg fitted function

# print(solutions)
# print(avgFittedFunction(solutions)[2])

# Training MSE plot validation
# print(MSE_toPlot(xpoints, solutions, solutionIndex, index))

#Scatter plot validation of test error
# ind = 0
# sol_Index = 12
# plt.scatter(dSTest[ind]['x'], dSTest[ind]['y'], color='r')
# plt.scatter(dSTest[ind]['x'], np.polyval(solutionSet[ind][sol_Index], dSTest[ind]['x']), color='y')
# plt.show()