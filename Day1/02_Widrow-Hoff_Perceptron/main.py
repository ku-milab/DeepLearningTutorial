import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Perceptron_Widrow_Hoff import *

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
    Task 2: Train and plot two Adaline (ADAptive LInear NEurons) with different learning rates
    '''
    #train_adaline_diff_lr(X, y)
    
    
    '''
    Task 3: Train and plot effect of data standardization
    '''
    #sX, classifier = data_standarization(X, y)
    
    
    '''
    Task 4: Ploy decision region of Adaline Perceptron with data standarization
    '''
    #plot_decision_region(sX, y, classifier)

    

def train_adaline_diff_lr(X,y):
    print("Two ADAptive LInear NEurons with different learning rates (0.01/0.0001) ")
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8,4))
    
    classifier1 = AdalineGD(eta=0.01, epochs=10)
    classifier1.fit(X, y)
    ax[0].plot(range(1,len(classifier1.errors)+1), np.log10(classifier1.errors), marker='o')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('log(Sum-squared-error)')
    ax[0].set_title('Adaline - Learning rate = 0.01')

    classifier2 = AdalineGD(eta=0.0001, epochs=10)
    classifier2.fit(X, y)
    ax[1].plot(range(1,len(classifier2.errors)+1), classifier2.errors, marker='o')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Sum-squared-error')
    ax[1].set_title('Adaline - Learning rate = 0.0001')
    plt.savefig('figure.png')
    elice_utils.send_image('figure.png')
    plt.clf()

def data_standarization(X, y):
    print("Benefitting from Standardization")
    sX = np.copy(X)
    sX[:,0] = (X[:,0]-X[:,0].mean())/X[:,0].std()
    sX[:,1] = (X[:,1]-X[:,1].mean())/X[:,1].std()

    classifier = AdalineGD(eta=0.01, epochs=10)
    classifier.fit(sX, y)
    plt.plot(range(1,len(classifier.errors)+1), classifier.errors, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Sum-squared-error')
    plt.savefig('figure.png')
    elice_utils.send_image('figure.png')
    plt.clf()
    
    return sX, classifier
    
def plot_decision_region(sX, y, classifier):
    print("Decision region of Adaline")
    plot_decision_regions(sX, y, classifier=classifier)
    plt.title('Adaline - Gradient Descent')
    plt.xlabel('Sepal Length [Standardized]')
    plt.ylabel('Petal Length [Standardized]')
    plt.legend(loc='upper left')
    plt.savefig('figure.png')
    elice_utils.send_image('figure.png')
    plt.clf()

if __name__=="__main__":
    main()
