import numpy as np
all_types = ['relu', 'sigmoid', 'tanh']


def relu(x):
    return max(0, x)


def diff_relu(x):
    if x < 0:
        return 0
    else:
        return 1


def sigmoid(x):
    return 1/(1+np.exp(-x))


def diff_sigmoid(x):
    return sigmoid(x)*(sigmoid(x)-1)


def tanh(x):
    return np.tanh(x)


def diff_tanh(x):
    return (2*np.cosh(x)/(np.cosh(2*x)+1))**2


def create_activation_function(activation_type):
    if activation_type=='sigmoid':
        activation_function = sigmoid
        activation_gradient = diff_sigmoid
    elif activation_type=='relu':
        activation_function = relu
        activation_gradient = diff_relu
    else:
        activation_function=tanh
        activation_gradient=diff_tanh
    return activation_function, activation_gradient


class NNLayer:
    def __init__(self, name, activation_type, level, size, f_in):
        self.__class_name__ = name
        assert activation_type in all_types
        self.type = activation_type
        self.level = level
        self.W = np.random.rand(f_in, size)
        self.X = np.array([])
        self.dX = 0
        self.activation, self.activation_gradient = create_activation_function(activation_type)

    def forward(self, X):
        self.X = X
        # return np.dot(self.X, self.W)
        a = map(self.activation, np.dot(self.X, self.W))
        return a
        # return self.activation(np.dot(self.X, self.W))

    def backward(self, dz):
        self.dX = dz * self.activation_gradient(self.X)

    def update(self):
        self.W += self.dX





