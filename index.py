from numpy import exp,array,random,dot


class NeuralNetwork():

    def __init__(self, n_input, n_hidden, n_output, learning_rate):

        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output

        self.learning_rate = learning_rate

        self.w_input = random.normal(0.0, pow(self.n_hidden, -0.5), (self.n_hidden, self.n_input))
        self.w_hidden = random.normal(0.0, pow(self.n_output, -0.5), (self.n_output, self.n_hidden))
        print(self.w_hidden)





if __name__ == "__main__":

    input_nodes = 784
    hidden_nodes = 200
    output_nodes = 10
    learning_rate = 0.1

    nn = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
