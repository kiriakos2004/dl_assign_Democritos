import torch
import torch.nn as nn
import torch.optim as optim

# Define training parameters
hidden_dim = 256
batch_size = 128
learning_rate = 0.0001
seq_len = 100
dropout_rate = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Generator(nn.Module):
    def __init__(self, seq_len, n_features, hidden_dim=hidden_dim, dropout_rate=dropout_rate):
        super(Generator, self).__init__()
        self.seq_len, self.n_features = seq_len, n_features
        self.hidden_dim = hidden_dim if hidden_dim else n_features
        self.rnn1 = nn.LSTM(input_size=n_features + hidden_dim, hidden_size=self.hidden_dim, num_layers=1, batch_first=True)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.rnn2 = nn.LSTM(input_size=self.hidden_dim, hidden_size=self.hidden_dim, num_layers=1, batch_first=True)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.output_layer = nn.Linear(self.hidden_dim, n_features)

    def forward(self, x, noise):
        x = torch.cat((x, noise), dim=-1)
        x, _ = self.rnn1(x)
        x = self.dropout1(x)
        x, (hidden_n, _) = self.rnn2(x)
        x = self.dropout2(x)
        x = self.output_layer(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, seq_len, n_features, hidden_dim=hidden_dim, dropout_rate=dropout_rate):
        super(Discriminator, self).__init__()
        self.seq_len, self.n_features = seq_len, n_features
        self.hidden_dim = hidden_dim if hidden_dim else n_features * 2
        self.rnn1 = nn.LSTM(input_size=n_features, hidden_size=self.hidden_dim, num_layers=1, batch_first=True)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.rnn2 = nn.LSTM(input_size=self.hidden_dim, hidden_size=self.hidden_dim, num_layers=1, batch_first=True)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.output_layer = nn.Linear(self.hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x, _ = self.rnn1(x)
        x = self.dropout1(x)
        x, _ = self.rnn2(x)
        x = self.dropout2(x)
        x = self.output_layer(x[:, -1, :])  # Take the last time step output for binary classification
        x = self.sigmoid(x)
        return x

