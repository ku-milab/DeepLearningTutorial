import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Perceptron_Widrow_Hoff import *

'''
Load Dataset
'''

print(">> Check Dataset")
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
df.tail()

'''
Plot dataset
'''
print("2D Plot of the IRIS Dataset")
y = df.iloc[0:100,4].values
y = np.where( y=='Iris-setosa', -1, 1 )
X = df.iloc[0:100, [0,2]].values
plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
plt.xlabel('Petal Length')
plt.ylabel('Sepal Length')
plt.legend(loc='upper left')
plt.show()
plt.clf()

'''
Two ADAptive LInear NEurons with different learning rates 
'''

print("Two ADAptive LInear NEurons with different learning rates (0.01/0.0001) ")
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8,4))
classifier1 = AdalineGD(eta=0.01, epochs=10)
classifier1.train(X, y)
ax[0].plot(range(1,len(classifier1.cost)+1), np.log10(classifier1.cost), marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Sum-squared-error)')
ax[0].set_title('Adaline - Learning rate = 0.01')

classifier2 = AdalineGD(eta=0.0001, epochs=10)
classifier2.train(X, y)
ax[1].plot(range(1,len(classifier2.cost)+1), classifier2.cost, marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Sum-squared-error')
ax[1].set_title('Adaline - Learning rate = 0.0001')
plt.show()
plt.clf()

'''
Effect of data standardization
'''
print("Benefitting from Standardization")
sX = np.copy(X)
sX[:,0] = (X[:,0]-X[:,0].mean())/X[:,0].std()
sX[:,1] = (X[:,1]-X[:,1].mean())/X[:,1].std()

classifier3 = AdalineGD(eta=0.01, epochs=10)
classifier3.train(sX, y)
plt.plot(range(1,len(classifier3.cost)+1), classifier3.cost, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Sum-squared-error')
plt.show()
plt.clf()

print("Decision region of Adaline")
plot_decision_regions(sX, y, classifier=classifier3)
plt.title('Adaline - Gradient Descent')
plt.xlabel('Sepal Length [Standardized]')
plt.ylabel('Petal Length [Standardized]')
plt.legend(loc='upper left')
plt.show()
plt.clf()

