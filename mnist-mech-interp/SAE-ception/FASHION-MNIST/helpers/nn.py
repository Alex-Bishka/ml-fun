from torch import nn

class NeuralNetwork(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()
        layer_size_by_pixels = 28*28
        
        self.flatten = nn.Flatten()
        
        # define layers separately to have access to each
        self.hidden_one = nn.Linear(layer_size_by_pixels, hidden_size)
        self.hidden_two = nn.Linear(hidden_size, hidden_size)
        self.hidden_three = nn.Linear(hidden_size, hidden_size)
        self.classification_layer = nn.Linear(hidden_size, 10)
        
        self.activation_function = nn.ReLU()

    def forward(self, x):
        x = self.flatten(x)

        # first hidden layer
        hidden_one_out = self.hidden_one(x)
        hidden_one_act = self.activation_function(hidden_one_out)

        # second hidden layer
        hidden_two_out = self.hidden_two(hidden_one_act)
        hidden_two_act = self.activation_function(hidden_two_out)

        # second hidden layer
        hidden_three_out = self.hidden_two(hidden_two_act)
        hidden_three_act = self.activation_function(hidden_three_out)

        # classification layer
        classification_out = self.classification_layer(hidden_three_act)
        
        return classification_out, hidden_one_act, hidden_two_act, hidden_three_act