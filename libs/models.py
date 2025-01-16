import torch
from torch import nn


# Define models
class NeuralNetwork(nn.Module):
    def __init__(self, h_size):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(7, h_size),
            nn.ReLU(),
            nn.Linear(h_size, h_size // 2),
            nn.ReLU(),
            nn.Linear(h_size // 2, h_size // 4),
            nn.ReLU(),
            nn.Linear(h_size // 4, 1)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class SimpleNN(nn.Module):
    def __init__(self, h_size):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(7, h_size),
            nn.ReLU(),
            nn.Linear(h_size, 1)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class NN(nn.Module):
    def __init__(self, h_size, d_rate):
        super().__init__()
        self.Flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(7, h_size),
            nn.BatchNorm1d(h_size),
            nn.ReLU(),
            nn.Dropout(d_rate),
            nn.Linear(h_size, h_size // 2),
            nn.BatchNorm1d(h_size // 2),
            nn.ReLU(),
            nn.Linear(h_size // 2, h_size // 4),
            nn.BatchNorm1d(h_size // 4),
            nn.ReLU(),
            nn.Linear(h_size // 4, 1)
        )

    def forward(self, x):
        x = self.Flatten(x)
        logit = self.linear_relu_stack(x)
        return logit


class LSTMNN(nn.Module):
    def __init__(self, hidden_size, num_layers, dropout_rate):
        super().__init__()
        self.lstm = nn.LSTM(7, hidden_size, num_layers, batch_first=True,
                            dropout=dropout_rate)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Take the output at the last timestep
        return out

class GRUNN(nn.Module):
    def __init__(self, hidden_size, num_layers, dropout_rate):
        super().__init__()
        self.GRU = nn.GRU(7, hidden_size, num_layers, batch_first=True,
                            dropout=dropout_rate)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Take the output at the last timestep
        return out



class CNN(nn.Module):
    def __init__(self, num_filters, kernel_size):
        super().__init__()
        self.conv1 = nn.Conv1d(7, num_filters, kernel_size)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        self.fc = nn.Linear(num_filters, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Change shape to (batch, channels, features)
        x = self.pool(self.relu(self.conv1(x)))
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x

