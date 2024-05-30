import torch
import torch.nn as nn
import torch.optim as optim

# Define training parameters
embedding_dim = 24
batch_size = 32
learning_rate = 0.001
seq_len = 10
dropout_rate = 0.1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim, dropout_rate):
        super(Encoder, self).__init__()
        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim
        self.rnn = nn.LSTM(input_size=n_features, hidden_size=self.hidden_dim, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.mean = nn.Linear(self.hidden_dim, embedding_dim)
        self.log_var = nn.Linear(self.hidden_dim, embedding_dim)

    def forward(self, x):
        batch_size = x.shape[0]
        x, (hidden_n, _) = self.rnn(x)
        x = self.dropout(x)
        hidden = hidden_n[-1].reshape(batch_size, self.hidden_dim)
        mean = self.mean(hidden)
        log_var = self.log_var(hidden)
        return mean, log_var

class Decoder(nn.Module):
    def __init__(self, seq_len, input_dim, n_features, dropout_rate):
        super(Decoder, self).__init__()
        self.seq_len, self.input_dim = seq_len, input_dim
        self.hidden_dim, self.n_features = 2 * input_dim, n_features
        self.rnn = nn.LSTM(input_size=input_dim, hidden_size=self.hidden_dim, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.output_layer = nn.Linear(self.hidden_dim, n_features)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.unsqueeze(1).repeat(1, self.seq_len, 1)
        x, _ = self.rnn(x)
        x = self.dropout(x)
        x = self.output_layer(x)
        return x

class RecurrentVAE(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim, dropout_rate, device):
        super(RecurrentVAE, self).__init__()
        self.encoder = Encoder(seq_len, n_features, embedding_dim, dropout_rate).to(device)
        self.decoder = Decoder(seq_len, embedding_dim, n_features, dropout_rate).to(device)

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x):
        mean, log_var = self.encoder(x)
        z = self.reparameterize(mean, log_var)
        decoded = self.decoder(z)
        return decoded, mean, log_var
