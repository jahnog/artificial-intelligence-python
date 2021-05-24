import numpy as np


class FourLayerModel:
    def __init__(self, input_features: int, hidden_layer1_neurons: int = 10, hidden_layer2_neurons: int = 5, output_layer_neurons: int = 1):
        """
        Arguments:
            input_features (int): The number of input features.
            hidden_layer1_neurons (int, optional): Number of Neurons for the first hidden layer. Defaults to 10.
            hidden_layer2_neurons (int, optional): Number of Neurons for the second hidden layer. Defaults to 5.
            output_layer_neurons (int, optional): Number of Neurons for the output layer. Defaults to 1.
        """
        np.random.seed(1)
        
        self.W1 = np.random.randn(hidden_layer1_neurons, input_features) * 0.01
        self.b1 = np.zeros((hidden_layer1_neurons, 1))

        self.W2 = np.random.randn(hidden_layer2_neurons, hidden_layer1_neurons) * 0.01
        self.b2 = np.zeros((hidden_layer2_neurons, 1))

        self.W3 = np.random.randn(output_layer_neurons, hidden_layer2_neurons) * 0.01
        self.b3 = np.zeros((output_layer_neurons, 1))

    def relu(self, Z):
        return np.maximum(Z, 0)

    def forward_propagation(self, X):
        """
        Argument:
        X -- Input data of shape (number of features, number of examples).

        Returns:
        The prediction of the model, of shape (1, number of examples)
        """

        # Calculate first hidden layer neurons
        self.Z1 = np.dot(self.W1, X) + self.b1
        # Calculate first hidden layer neuron activations
        self.A1 = self.relu(self.Z1)

        # Calculate second hidden layer neurons
        self.Z2 = np.dot(self.W2, self.A1) + self.b2
        # Calculate second hidden layer neuron activations
        self.A2 = self.relu(self.Z2)

        # Calculate output layer neurons
        self.Z3 = np.dot(self.W3, self.A2) + self.b3
        # Calculate output layer neuron activations
        self.A3 = self.Z3

        return self.A3

    def calculate_cost(self, prediction, Y):
        """
        Arguments:
        prediction -- The output of last activation, of shape (1, number of examples)
        Y -- "true" labels vector of shape (1, number of examples)

        Returns:
        cost -- computes the Mean Squared Error
        """

        return np.mean(np.square(Y - prediction))

    def relu_dev(self, X):
        return np.where(X > 0, 1, 0)

    def backward_propagation(self, X, Y):
        """
        Arguments:
        X -- Input data of shape (number of features, number of examples).
        Y -- "true" labels vector of shape (1, number of examples)
        """

        # Number of examples
        m = X.shape[1]

        # Mean Suared Error derivative
        self.dZ3 = -2 * (Y - self.A3)

        # Output layer derivatives
        self.dW3 = np.dot(self.dZ3, self.A2.T) / m
        self.db3 = np.sum(self.dZ3, axis=1, keepdims=True) / m

        # Second hidden layer derivatives
        self.dZ2 = np.dot(self.W3.T, self.dZ3) * self.relu_dev(self.A2)
        self.dW2 = np.dot(self.dZ2, self.A1.T) / m
        self.db2 = np.sum(self.dZ2, axis=1, keepdims=True) / m

        # First hidden layer derivatives
        self.dZ1 = np.dot(self.W2.T, self.dZ2) * self.relu_dev(self.A1)
        self.dW1 = np.dot(self.dZ1, X.T) / m
        self.db1 = np.sum(self.dZ1, axis=1, keepdims=True) / m

    def update_weights(self, learning_rate=0.01):
        """
        Argument:
        learning_rate -- The learning rate to apply in the weights update.
        """
        self.W1 = self.W1 - learning_rate * self.dW1
        self.b1 = self.b1 - learning_rate * self.db1
        self.W2 = self.W2 - learning_rate * self.dW2
        self.b2 = self.b2 - learning_rate * self.db2
        self.W3 = self.W3 - learning_rate * self.dW3
        self.b3 = self.b3 - learning_rate * self.db3
