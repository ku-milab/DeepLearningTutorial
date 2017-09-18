import numpy as np
import matplotlib.pyplot as plt
import NeuralNetwork as nn
from sklearn.datasets import *
from sklearn.cross_validation import train_test_split
import sys
import elice_utils


num_units = np.array([2, 5, 2]) #Number of units per each layer
eta = 0.01  # Learning rate
epochs = 100 # Number of epochs
act_fun = 'tanh' # Activation function, (tanh, sigmoid)
seed = 10  #Random seed for data creation, validation set chooser
data = 0  #Data maker, 0-moons, 1-circle, 2-blobs, 3-guassian quantiles
n_samples = 500  #Number of data to use

def main():
    '''
    Read and check data
    '''
    X_train, X_test, y_train, y_test = load_data()

    '''
    Train Neural Network
    '''
    train_nn(X_train, y_train, X_test, y_test)


def load_data():
    # Load data from scikit-learn
    if data==1:
        X, y = make_circles(n_samples=n_samples, random_state=seed, noise=0.1)
    elif data==2:
        X, y = make_blobs(n_samples=n_samples, random_state=seed)
    elif data==3:
        X,y  = make_gaussian_quantiles(n_samples=n_samples, random_state=seed, n_classes=2)
    else:
        X, y = make_moons(n_samples=n_samples, random_state=seed, noise=0.1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)

    y0 = np.where(y_train == 0)
    y1 = np.where(y_train == 1)
    plt.scatter(X_train[y0, 0], X_train[y0, 1], color='red', marker='o')
    plt.scatter(X_train[y1, 0], X_train[y1, 1], color='blue', marker='x')
    plt.savefig('figure.png')
    elice_utils.send_image('figure.png')
    plt.clf()
    return X_train, X_test, y_train, y_test


def train_nn(X_train, y_train, X_test, y_test):

    model = nn.NeuralNetwork(num_units=num_units, eta=eta, epochs=epochs, activation_function=act_fun,
                             minibatch_size=X_train.shape[0], bShuffle=True, random_state=None, verbose=True)
    model.initialize_weights()
    model.train(X_train, y_train, X_test, y_test)
    predicted = model.predict(X_test)
    labels = np.argmax(predicted, axis=1)

    _, axes = plt.subplots(nrows=1, ncols=2)
    y0 = np.where(y_test == 0)
    y1 = np.where(y_test == 1)
    axes[0].scatter(X_test[y0, 0], X_test[y0, 1], color='red', marker='o')
    axes[0].scatter(X_test[y1, 0], X_test[y1, 1], color='blue', marker='x')


    y0 = np.where(labels == 0)
    y1 = np.where(labels == 1)
    axes[1].scatter(X_test[y0, 0], X_test[y0, 1], color='red', marker='o')
    axes[1].scatter(X_test[y1, 0], X_test[y1, 1], color='blue', marker='x')

    plt.savefig('figure.png')
    elice_utils.send_image('figure.png')
    plt.clf()
    print(np.sum(labels == y_test) / float(len(y_test)))


if __name__ == "__main__":
    main()
