import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class NeuralNetwork(object):
    """ 
        <Network parameters>
        weights[0]: {array-like}, shape=[num_hidden_units_layer, num_features+1]
        weights[1]: {array-like}, shape=[num_classes, num_hidden_units_layer+1]

        <Auxiliary variables necessary for backpropagation>
        delta[0]: {array-like}, shape=[num_hidden_units_layer]
        delta[1]: {array-like}, shape=[num_classes]

        net_input[0]: {array-like}, shape=[num_hidden_units_layer]
        net_input[1]: {array-like}, shape=[num_classes]
    """

    def __init__(self, num_units, eta=0.001, epochs=100, activation_function='sigmoid', minibatch_size=16, bShuffle=True, random_state=None, verbose=False):
        self.num_units = num_units
        self.eta = eta
        self.epochs = epochs
        self.weights_initialized = False
        self.bShuffle = bShuffle
        self.minibatch_size = minibatch_size
        self.activation_function = activation_function
        self.verbose = verbose
        self.weights = []
        self.gradient = []
        self.delta = []
        self.net_value = []
        self.activation_value = []
        self.training_cost = []
        self.training_error = []
        self.validation_error = []
        if random_state:
            np.random.seed(random_state)

    def buffer_clear(self):
        self.weights = []
        self.gradient = []
        self.delta = []
        self.net_value = []
        self.activation_value = []
        self.training_cost = []
        self.training_error = []
        self.validation_error = []

    def initialize_weights(self):
        self.buffer_clear()
        self.weights_initialized = False

        # Random parameter initialization
        for i in range(len(self.num_units)-1):
            self.weights.append(np.random.normal(0, 1.0, (self.num_units[i+1], 1 + self.num_units[i])))

        # Auxiliary variables initialization
        for i in range(len(self.num_units)):
            self.net_value.append(np.zeros(self.num_units[i]))
            self.activation_value.append( np.zeros(self.num_units[i]))
            self.delta.append(np.zeros(self.num_units[i] + 1))

        self.weights_initialized = True

    def compute_loss(self, y):
        # cross-entropy (multi-class)
        return -y.dot(np.log(self.activation_value[-1]))

    def activation(self, layer, input_value):
        W = self.weights[layer]
        aug_value = self.vector_augmentation(input_value)
        net_value = np.dot(W, aug_value.T)

        if layer == len(self.weights)-1:
            return net_value, self.softmax(net_value)
        else:
            if self.activation_function=='sigmoid':
                return net_value, self.sigmoid(net_value)
            elif self.activation_function=='tanh':
                return net_value, self.tanh(net_value)

    def forward(self, x):
        self.activation_value[0] = x
        for layer in range(len(self.num_units)-1):
            self.net_value[layer+1], self.activation_value[layer+1] = self.activation(layer, self.activation_value[layer])

    # logistic sigmoid for binary classification
    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def tanh(self, z):
        return np.tanh(z)

    def softmax(self, z):
        exp_z = np.exp(z)
        return exp_z / exp_z.sum()

    def derivative_activation(self, z):
        if self.activation_function == 'sigmoid':
            return self.derivative_sigmoid(z)
        elif self.activation_function == 'tanh':
            return self.derivative_tanh(z)

    def derivative_sigmoid(self, z):
        sigmoid_value = self.sigmoid(z)
        return sigmoid_value * (1 - sigmoid_value)

    def derivative_tanh(self, z):
        tanh_value = self.tanh(z)
        return 1-(tanh_value**2)

    def predict(self, X):
        predicted = []
        for i in range(X.shape[0]):
            self.forward(X[i])
            predicted.append(self.get_predicted_label())
        return predicted

    def get_predicted_label(self):
        idx = np.argmax( self.activation_value[-1] )
        label = np.zeros(self.num_units[-1])
        label[idx] = 1.0
        return label

    def train(self, X_train, y_train, X_val, y_val):
        X_data, y_data = X_train.copy(), self.one_hot_coding(y_train)
        X_val_data, y_val_data = X_val.copy(), y_val.copy()

        if self.weights_initialized==False:
            self.initialize_weights()

        for epoch in range(1, self.epochs+1):
            if self.bShuffle:
                X_data, y_data = self.shuffle(X_data, y_data)
            costs = []

            for i in range(0, X_data.shape[0], self.minibatch_size):
                if (i + self.minibatch_size) <= X_data.shape[0]:
                    miniX = X_data[i:i + self.minibatch_size]
                    miniY = y_data[i:i + self.minibatch_size]
                else:
                    miniX = X_data[i:]
                    miniY = y_data[i:]

                costs = []
                batch_gradient = []
                for xi, target in zip(miniX, miniY):
                    self.forward(xi)
                    cost = self.compute_loss(target)
                    costs.append(cost)
                    gradient = self.compute_gradient(target)
                    batch_gradient.append(gradient)
                acc_gradient = np.sum(batch_gradient, axis=0)
                self.update_weights(acc_gradient)

                avg_cost = np.average(costs)
                self.training_cost.append(avg_cost)

                train_predicted = self.predict(X_train)
                train_labels = np.argmax(train_predicted, axis=1)
                self.training_error.append(np.sum(train_labels != y_train) / float(len(y_train)))

                val_predicted = self.predict(X_val_data)
                val_labels = np.argmax(val_predicted, axis=1)
                self.validation_error.append(np.sum(val_labels != y_val_data) / float(len(y_val_data)))


            if self.verbose==True:
                # print avg_cost
                print('Epoch {} out of {} is done...'.format(epoch, self.epochs))

    def compute_gradient(self, y):
        for i in range(len(self.delta)-1, -1, -1):
            if i==len(self.delta)-1:
                self.delta[-1] = self.vector_augmentation(y-self.activation_value[-1], how='row')
            else:
                net_value = self.vector_augmentation(self.net_value[i], how='row')

                # delta_j = h'(a_j)*(sum_k{w_k*delta_k})

                incoming_message = np.dot(self.weights[i].T, self.delta[i+1][1:])
                self.delta[i] = np.multiply(incoming_message, self.derivative_activation(net_value))

        gradient = []
        for i in range(len(self.weights)):
            activation_value = self.vector_augmentation(self.activation_value[i], how='column')
            # if i==len(self.weights)-1:
            #     gradient.append(np.outer(self.delta[i+1], activation_value))
            # else:
            gradient.append(np.outer(self.delta[i+1][1:], activation_value))
        return gradient

    def update_weights(self, acc_gradient):
        for i in range(len(self.weights)):
            self.weights[i] += self.eta * acc_gradient[i]

    def vector_augmentation(self, X, how='column'):
        if X.ndim == 1:
            augX = np.ones(1 + len(X))
            augX[1:] = X
        else:
            if how == 'column':
                augX = np.ones((X.shape[0], X.shape[1]+1))
                augX[:, 1:] = X
            elif how == 'row':
                augX = np.ones((X.shape[0]+1, X.shape[1]))
                augX[1:, :] = X
            else:
                raise AttributeError('how should be either (column) or (row)')
        return augX

    def shuffle(self, X, y):
        r = np.random.permutation(len(y))
        return X[r], y[r]

    def one_hot_coding(self, y):
        classes = np.unique(y)
        one_hot_code = np.zeros((y.shape[0], len(classes)))
        for idx, val in enumerate(y):
            one_hot_code[idx, val] = 1.0
        return one_hot_code

    def plot_decision_regions(self, X, y, resolution=0.02):
        if X.shape[1] != 2:
            print('plot_decision_regions function is only valid for 2d space...')
        else:
            # setup marker generator and color map
            markers = ('s', 'x', 'o', '^', 'v')
            colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
            cmap = ListedColormap(colors[:len(np.unique(y))])

            # plot the decision surface
            x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

            xx1, xx2 = np.meshgrid( np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution) )
            Z = self.predict( np.array([xx1.ravel(), xx2.ravel()]).T )
            Z = np.argmax( Z, axis=1 )
            Z = Z.reshape( xx1.shape )
            plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
            plt.xlim(xx1.min(), xx1.max())
            plt.ylim(xx2.min(), xx2.max())

            # plot class samples
            for idx, c1 in enumerate(np.unique(y)):
                plt.scatter(x=X[y == c1, 0], y=X[y == c1, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=c1)