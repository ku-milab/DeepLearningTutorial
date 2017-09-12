import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Perceptron_Rosenblatt import *

def main():
    '''
    Load Dataset
    '''
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

    '''
    Check Dataset
    '''
    y = df.iloc[0:100,4].values
    y = np.where( y== 'Iris-setosa', -1, 1 )
    X = df.iloc[0:100, [0,2]].values
    plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
    plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
    plt.xlabel('Petal Length')
    plt.ylabel('Sepal Length')
    plt.legend(loc='upper left')
    plt.show()

    '''
    Training Perceptron
    '''
    classifier = Perceptron(eta=0.1, epochs=10)
    classifier.fit(X, y)
    plt.plot(range(1,len(classifier.errors)+1), classifier.errors, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Number of misclassifications')
    plt.show()

    plot_decision_regions(X, y, classifier=classifier)
    plt.xlabel('Sepal Length [cm]')
    plt.ylabel('Petal Length [cm]')
    plt.legend(loc='upper left')
    plt.show()

    '''
    Plot Decision Regions
    '''
    plot_decision_regions(X, y, classifier=classifier)
    plt.xlabel('Sepal Length [cm]')
    plt.ylabel('Petal Length [cm]')
    plt.legend(loc='upper left')
    plt.show()

    return

if __name__=="__main__":
    main()