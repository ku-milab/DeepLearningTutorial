import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons
from sklearn.cross_validation import train_test_split

# Load data from scikit-learn
X, y = make_moons(n_samples=500, random_state=10, noise=0.1)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10)

# y0 = np.where( y_train==0 )
# y1 = np.where( y_train==1 )
# plt.scatter( X_train[y0,0], X_train[y0,1], color='red', marker='o' )
# plt.scatter( X_train[y1,0], X_train[y1,1], color='blue', marker='x' )
# plt.show()

import NeuralNetwork as nn
# from .NeuralNetwork import NeuralNetwork as nn

# two-layer(input-hidden-output) neural network
num_units = [2, 5, 2]
model = nn.NeuralNetwork( num_units, eta=0.01, epochs=100, activation_function='tanh', minibatch_size=X_train.shape[0], bShuffle=True, random_state=None, verbose=False )
model.initialize_weights()
model.train( X_train, y_train, X_test, y_test )

predicted = model.predict(X_test)
labels = np.argmax(predicted, axis=1)
np.sum(labels==y_test)/float(len(y_test))

