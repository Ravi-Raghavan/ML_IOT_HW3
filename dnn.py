import numpy as np

class SimpleNN:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.W1, self.b1, self.W2, self.b2 = self.initialize_parameters()

    def initialize_parameters(self):
        # Initialize weights and biases
        # TODO: Initialize W1, b1, W2, and b2
        
        #Shape of W1: (hidden_dim, input_dim)
        W1 = np.random.randn(self.hidden_dim, self.input_dim) * np.sqrt(1 / (self.input_dim + self.hidden_dim))
        
        #Shape of b1: (hidden_dim)
        b1 = np.random.randn(self.hidden_dim, 1) * np.sqrt(1 / (self.input_dim + self.hidden_dim))
        
        #Shape of W2: (output_dim, hidden_dim)
        W2 = np.random.randn(self.output_dim, self.hidden_dim) * np.sqrt(1 / (self.hidden_dim + self.output_dim))
        
        #Shape of b2: (output_dim)
        b2 = np.random.randn(self.output_dim, 1) * np.sqrt(1 / (self.hidden_dim + self.output_dim))
        
        return W1, b1, W2, b2

    def sigmoid(self, Z):
        # Sigmoid activation function
        # TODO: Implement sigmoid activation
        return 1 / (1 + np.exp(-1 * Z))

    def forward_propagation(self, X):
        # Forward propagation
        # TODO: Implement forward propagation
        
        Z1 = self.W1 @ X + np.tile(self.b1, reps = (1, X.shape[1]))
        A1 = self.sigmoid(Z1)
        
        Z2 = self.W2 @ A1 + np.tile(self.b2, reps = (1, A1.shape[1]))
        A2 = self.sigmoid(Z2)
        
        return Z1, A1, Z2, A2

    def compute_cost(self, A2, Y):
        # Compute the cost
        # TODO: Implement cost computation
        
        #Clip values to avoid log(0) :) 
        epsilon = 1e-20
        A2_clipped = np.clip(A2, epsilon, 1 - epsilon)
        
        loss = - (Y * np.log(A2_clipped) + (1 - Y) * np.log(1 - A2_clipped))
        average_loss = np.mean(loss)
        
        return average_loss

    def backpropagation(self, X, Y, Z1, A1, Z2, A2):
        grads = {}
        m = X.shape[1]

        # Compute gradients
        dZ2 = A2 - Y
        dW2 = np.dot(dZ2, A1.T) / m
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m
        dZ1 = np.dot(self.W2.T, dZ2) * (A1 * (1 - A1))
        dW1 = np.dot(dZ1, X.T) / m
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m

        grads['dW1'] = dW1
        grads['db1'] = db1
        grads['dW2'] = dW2
        grads['db2'] = db2

        return grads

    def update_parameters(self, grads, learning_rate):
        # Update weights and biases
        self.W1 -= learning_rate * grads['dW1']
        self.b1 -= learning_rate * grads['db1']
        self.W2 -= learning_rate * grads['dW2']
        self.b2 -= learning_rate * grads['db2']


    def train(self, X, Y, learning_rate, num_iterations):
        for i in range(num_iterations):
            Z1, A1, Z2, A2 = self.forward_propagation(X)
            cost = self.compute_cost(A2, Y)
            grads = self.backpropagation(X, Y, Z1, A1, Z2, A2)
            self.update_parameters(grads, learning_rate)
            if i % 100 == 0:
                print(f"Iteration {i}, Cost: {cost}")

if __name__ == "__main__":
    # Generate synthetic data
    X = np.random.rand(2, 500)
    Y = np.random.randint(0, 2, size=(1, 500))

    # Initialize the neural network
    nn = SimpleNN(2, 4, 1)

    # Train the neural network
    nn.train(X, Y, learning_rate=0.01, num_iterations=1000)
