import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


class Perceptron(object):
    """
    Parameters: eta, epochs
    Attributes: weights, errors
    """

    def __init__(self, eta=0.01, epochs=10):
        self.eta = eta
        self.epochs = epochs

    def fit(self, X, y):
        """
        X: {array-like}, shape=[num_samples, num_features]
        y: {array-like}, shape=[num_samples]
        TASK 2: Implement weight update, i.e., fit
        """
        self.weights = np.zeros(1 + X.shape[1])  # self.weights[0] for bias
        self.errors = []

        for _ in range(self.epochs):
            errors = 0
            for xi, target in zip(X, y):
                '''***Task 2 : Implement weight update part of method fit(self, X, y)***'''
            '''How much we update Perceptron is decided by 
            learning rate * difference between label and model prediction'''
            # update =

            # Update weights of Perceptron
            # self.weights[1:] +=

            # Update bias of Perceptron
            # self.weights[0] +=

            # increase error variable by 1 if the perceptron's prediction is wrong
            # errors +=
        self.errors.append(errors)

        return self.errors


    def net_input(self, X):
        '''
        :param X: Input to the perceptron, shape=[num_data]
        :return: Output of the perceptron, shape=[num_data]
        TASK 1: Implement perceptron's output z
        '''
        z = None
        # TODO
        return z


    def predict(self, X):
        '''
        :param X: Input to the model, shape=[num_data]
        :return: Prediction of the model. Correct 1, incorrect -1, shape=[num_data]
        TASK 2: Implement prediction of model
        '''
        y_star = None
        # TODO
        return y_star


def plot_decision_regions(X, y, classifier, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, c1 in enumerate(np.unique(y)):
        plt.scatter(x=X[y == c1, 0], y=X[y == c1, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=c1)