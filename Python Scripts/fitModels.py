import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import matplotlib.pyplot as plt

ymin = -math.pi
ymax = math.pi

fileName = "C:/Users/deifen/Documents/Projects/Bias and overfitting trade offs/Project/SampleData/sampledata 1"
fileName2 = "C:/Users/deifen/Documents/Projects/Bias and overfitting trade offs/Project/SampleData/sampledata 2"
fileName3 = "C:/Users/deifen/Documents/Projects/Bias and overfitting trade offs/Project/SampleData/sampledata 3"


df=pd.read_csv(fileName, sep=',')
df2=pd.read_csv(fileName2, sep=',')
df3=pd.read_csv(fileName3, sep=',')

vals=df.values
vals2=df2.values
vals3=df3.values


x1 = np.array(vals3[:,0])
y1 = np.array(vals3[:,1])

x_fit = np.linspace(-math.pi,math.pi, 10)

p1 = np.polyfit(x1, y1, 1)
p2 = np.polyfit(x1, y1, 2)
p3 = np.polyfit(x1, y1, 3)
p4 = np.polyfit(x1, y1, 19)

# plt.plot(x1, y1, 'y*')
# plt.plot(x1, np.polyval(p4, x1), label='quadratic fit', linestyle='--')
# plt.show()

# Feed data into pyplot.
polynomial1 = poly1d(p1)
polynomial2 = poly1d(p2)
polynomial3 = poly1d(p3)
polynomial4 = poly1d(p4)

xpoints = np.linspace(-math.pi, math.pi, 100)
axes = plt.gca()
axes.set_ylim([ymin,ymax])

plt.plot(x1,y1,'x',xpoints,polynomial1(xpoints),'r-')
plt.plot(xpoints,polynomial2(xpoints),'b-')
plt.plot(xpoints,polynomial3(xpoints),'g-')
plt.plot(xpoints,polynomial4(xpoints),'y-')
plt.show()


print("Mean squared error: %.2f"
      % np.mean((np.polyval(p4, x1) - y1) ** 2))
print(p3)