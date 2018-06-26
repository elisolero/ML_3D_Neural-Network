from numpy import exp,array,random,dot


class NeuralNetwork():

    def __init__(self, n_input, n_hidden, n_output, learning_rate):

        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output

        self.learning_rate = learning_rate

        self.w_input = random.normal(0.0, pow(self.n_hidden, -0.5), (self.n_hidden, self.n_input))
        self.w_hidden = random.normal(0.0, pow(self.n_output, -0.5), (self.n_output, self.n_hidden))
        # print(self.w_hidden)

    def train(self, inputs_list, targets_list):
        inputs = array(inputs_list, ndmin=2).T
        targets = array(targets_list, ndmin=2).T

        hidden_inputs = dot(self.w_hidden, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = dot(self.w_input, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        output_errors = targets - final_outputs
        hidden_errors = dot(self.w_hidden.T, output_errors)

        self.w_input += self.learning_rate * dot(output_errors * final_outputs * (1.0 - final_outputs), hidden_outputs.T)
        self.w_hidden += self.learning_rate * dot(hidden_errors * hidden_outputs * (1.0 - hidden_outputs), inputs.T)

    # def __init__(self):
    #
    #     random.seed(1)  # get random int 1-10
    #     self.weights = 2 * random.random((3, 1)) - 1 #render weights
        # print(self.weights)


    def trainModel(self,training_set_inputs, training_set_outputs, number_of_training_iterations):
        # print(training_set_inputs)

        for iteration in xrange(number_of_training_iterations):

            output = self.think(training_set_inputs)

            # Calculate the error (The difference between the desired output
            # and the predicted output).
            error = training_set_outputs - output #t

            # Multiply the error by the input and again by the gradient of the Sigmoid curve.
            # This means less confident weights are adjusted more.
            # This means inputs, which are zero, do not cause changes to the weights.
            adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))


            self.weights += adjustment


    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))


    def think(self, inputs):
        # Pass inputs through our neural network (our single neuron).
        dotForSigmoid = dot(inputs, self.weights)
        return self.__sigmoid(dotForSigmoid)

    # The derivative of the Sigmoid function.
    # This is the gradient of the Sigmoid curve.
    # It indicates how confident we are about the existing weight.
    def __sigmoid_derivative(self, x):
        return x * (1 - x)


if __name__ == "__main__":

    input_nodes = 784
    hidden_nodes = 200
    output_nodes = 10
    learning_rate = 0.1

    nn = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    # NeuralNetwork = NeuralNetwork()

    # training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    # training_set_outputs = array([[0, 1, 1, 0]]).T
    #
    # # Train the neural network using a training set.
    # # Do it 10,000 times and make small adjustments each time.
    # NeuralNetwork.trainModel(training_set_inputs, training_set_outputs, 10000)

    # print NeuralNetwork.think(array([1, 0, 0]))

    # print "New synaptic weights after training: "
    # print NeuralNetwork.weights
    #
    # # Test the neural network with a new situation.
    # print "Considering new situation [1, 0, 0] -> ?: "
    # print NeuralNetwork.think(array([1, 0, 0]))
