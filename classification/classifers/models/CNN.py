from torch import nn


class CNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_label=2):
        super(CNNModel, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, kernel_size=3, stride=2, padding=1), # 12
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1), # 12
            nn.SiLU(),
            nn.Conv1d(hidden_size, hidden_size*2, kernel_size=3, stride=2, padding=1), # 6
            nn.Conv1d(hidden_size*2, hidden_size*2, kernel_size=3, stride=1, padding=1), # 6
            nn.SiLU(),
            nn.Conv1d(hidden_size*2, hidden_size*4, kernel_size=3, stride=2, padding=1), # 3
            nn.Conv1d(hidden_size*4, hidden_size*4, kernel_size=3, stride=1, padding=1), # 3
            nn.SiLU(),
            nn.AdaptiveAvgPool1d((1,)),
            nn.Flatten(),
        )

        self.fc_linear = nn.Linear(hidden_size*4, num_label)

    def forward(self, x):
        x = x[:, None, :]
        x = self.layers(x)

        x = self.fc_linear(x)

        return x