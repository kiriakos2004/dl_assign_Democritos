import torch
import torch.nn as nn
import torch.optim as optim

# Define training parameters
embedding_dim = 24
batch_size = 128
learning_rate = 0.001
seq_len = 10
dropout_rate = 0.2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=embedding_dim, dropout_rate=dropout_rate):
        super(Encoder, self).__init__()
        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim, self.hidden_dim = int(embedding_dim), int(2 * embedding_dim)
        self.rnn1 = nn.LSTM(input_size=n_features, hidden_size=self.hidden_dim, num_layers=1, batch_first=True)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.rnn2 = nn.LSTM(input_size=self.hidden_dim, hidden_size=self.embedding_dim, num_layers=1, batch_first=True)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        batch_size = x.shape[0]
        x, _ = self.rnn1(x)
        x = self.dropout1(x)
        x, (hidden_n, _) = self.rnn2(x)
        x = self.dropout2(x)
        encoded = hidden_n[-1].reshape(batch_size, self.embedding_dim)  # Take the last hidden state
        return encoded

class Decoder(nn.Module):
    def __init__(self, seq_len, input_dim, n_features, dropout_rate=dropout_rate):
        super(Decoder, self).__init__()
        self.seq_len, self.input_dim = seq_len, input_dim
        self.hidden_dim, self.n_features = int(2 * input_dim), n_features
        self.rnn1 = nn.LSTM(input_size=input_dim, hidden_size=input_dim, num_layers=1, batch_first=True)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.rnn2 = nn.LSTM(input_size=input_dim, hidden_size=self.hidden_dim, num_layers=1, batch_first=True)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.output_layer = nn.Linear(self.hidden_dim, n_features)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.unsqueeze(1).repeat(1, self.seq_len, 1)
        x, _ = self.rnn1(x)
        x = self.dropout1(x)
        x, _ = self.rnn2(x)
        x = self.dropout2(x)
        x = self.output_layer(x)
        return x

class RecurrentAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=embedding_dim, dropout_rate=dropout_rate, device=device):
        super(RecurrentAutoencoder, self).__init__()
        self.encoder = Encoder(seq_len, n_features, embedding_dim, dropout_rate).to(device)
        self.decoder = Decoder(seq_len, embedding_dim, n_features, dropout_rate).to(device)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded