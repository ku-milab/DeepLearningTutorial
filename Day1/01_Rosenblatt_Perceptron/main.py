import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Perceptron_Rosenblatt import *

import elice_utils

def main():

    '''
    Task 4: Check and Load Dataset
    '''
    #X,y = check_and_load_dataset()
    
    '''
    Task 5: Training Perceptron
    '''
    #classifier = training_Perceptron(X,y,0.1,10)

    '''
    Task 6: Plot Perceptron's Decision regions
    '''
    #plot_decision_region(X,y,classifier)
    
    
def check_and_load_dataset():
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
    print("IRIS DATASET\n")
    print(df)
    print("2D Plot of the IRIS Dataset")
    y = df.iloc[0:100,4].values
    y = np.where( y== 'Iris-setosa', -1, 1 )
    X = df.iloc[0:100, [0,2]].values
    plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
    plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
    plt.xlabel('Petal Length')
    plt.ylabel('Sepal Length')
    plt.legend(loc='upper left')
    plt.savefig('figure.png')
    elice_utils.send_image('figure.png')
    plt.clf()
    
    return X,y

def training_Perceptron(X,y,learning_rate,num_epochs):
    print("Plot # of misclassification vs Epoch")
    classifier = Perceptron(eta=learning_rate, epochs=num_epochs)
    classifier.fit(X, y)
    plt.plot(range(1,len(classifier.errors)+1), classifier.errors, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Number of misclassifications')
    plt.savefig('figure.png')
    elice_utils.send_image('figure.png')
    plt.clf()
    
    return classifier

def plot_decision_region(X,y,classifier):
    print("2D Plot of Perceptron's Decision Boundary")
    plot_decision_regions(X, y, classifier=classifier)
    plt.xlabel('Sepal Length [cm]')
    plt.ylabel('Petal Length [cm]')
    plt.legend(loc='upper left')
    plt.savefig('figure.png')
    elice_utils.send_image('figure.png')
    plt.clf()
    return
    
if __name__=="__main__":
    main()