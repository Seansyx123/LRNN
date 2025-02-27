from torch import nn


class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=512, num_label=2):
        super(LSTMModel, self).__init__()

        self.embedding = nn.Linear(input_size, hidden_size)

        self.layers = nn.LSTM(hidden_size, hidden_size, num_layers=4, batch_first=True)

        self.fc_linear = nn.Linear(hidden_size, num_label)

    def forward(self, x):
        x = x[:, :, None]
        x = self.embedding(x)
        x, _ = self.layers(x)
        x = x[:, -1]
        x = self.fc_linear(x)

        return x