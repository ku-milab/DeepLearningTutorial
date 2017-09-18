import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Perceptron_Widrow_Hoff_Stochastic import *

import elice_utils

def main():
    '''
    Load Dataset
    '''
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
    
    y = df.iloc[0:100,4].values
    y = np.where( y== 'Iris-setosa', -1, 1 )
    X = df.iloc[0:100, [0,2]].values
    
    
    '''
    Task 2: Train and plot SGD with different minibatch size
    '''
    
    train_SGD_diff_minibatch(X, y, 1)
    train_SGD_diff_minibatch(X, y, 10)

def train_SGD_diff_minibatch(X, y, minibatch_size):
    # Benefitting from Standardization
    sX = np.copy(X)
    sX[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
    sX[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

    classifier1 = AdalineSGD(eta=0.01, epochs=15, random_state=1, minibatch_size=minibatch_size)
    classifier1.fit_SGD(sX, y)

    plot_decision_regions(sX, y, classifier=classifier1)
    plt.title('Adaline - Stochastic Gradient Descent')
    plt.xlabel('Sepal Length [Standardized]')
    plt.ylabel('Petal Length [Standardized]')
    plt.legend(loc='upper left')
    plt.savefig('figure.png')
    elice_utils.send_image('figure.png')
    plt.clf()

    plt.plot(range(1, len(classifier1.errors) + 1), classifier1.errors, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Average Cost')
    plt.savefig('figure.png')
    elice_utils.send_image('figure.png')
    plt.clf()
    

if __name__=="__main__":
    main()
