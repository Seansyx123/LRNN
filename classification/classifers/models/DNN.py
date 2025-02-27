from torch import nn


class DNNModel(nn.Module):
    def __init__(self, input_size=21, hidden_size=512, num_label=2):
        super(DNNModel, self).__init__()

        self.embedding = nn.Linear(input_size, hidden_size)

        self.layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size*4),
            nn.SiLU(),
            nn.Linear(hidden_size*4, hidden_size)
        )

        self.fc_linear = nn.Linear(hidden_size, num_label)

    def forward(self, x):

        x = self.embedding(x)
        x = self.layers(x)
        x = self.fc_linear(x)

        return x