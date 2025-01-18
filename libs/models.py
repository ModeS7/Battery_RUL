import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


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
            nn.BatchNorm1d(h_size),
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
            nn.Linear(h_size, 1)
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
    def __init__(self, num_channels, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(1, num_channels, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool(self.relu(self.conv1(x)))
        flattened_size = x.shape[1] * x.shape[2]
        if not hasattr(self, 'fc'):
            self.fc = nn.Linear(flattened_size, 1).to(x.device)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x


class TNN(nn.Module): #Transformer Neural Network
    def __init__(self, d_model, nhead, num_layers):
        super().__init__()
        self.embedding = nn.Linear(7, d_model)
        encoder_layer = TransformerEncoderLayer(d_model, nhead)
        self.transformer = TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.embedding(x)
        x = self.transformer(x)
        x = self.fc(x[:, -1, :])  # Use the last timestep
        return x


class ANN(nn.Module): #Autoencoder Neural Network
    def __init__(self, latent_size):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(7, latent_size),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 1),
            nn.ReLU()
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed  # Only return the reconstructed output
