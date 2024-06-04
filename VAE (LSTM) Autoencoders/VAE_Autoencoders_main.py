import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from VAE_LSTM_dropout import RecurrentVAE
import VAE_LSTM_dropout
from VAE_LSTM_dropout_one_layer import RecurrentVAE_one_layer
import VAE_LSTM_dropout_one_layer
import matplotlib.pyplot as plt
import math
from scipy import stats

num_epochs = 200
patience = 10
weight_decay = 0

#choose parameters specific to model

dropout_rate = VAE_LSTM_dropout_one_layer.dropout_rate
embedding_dim = VAE_LSTM_dropout_one_layer.embedding_dim
learning_rate = VAE_LSTM_dropout_one_layer.learning_rate
batch_size = VAE_LSTM_dropout_one_layer.batch_size
seq_len = VAE_LSTM_dropout_one_layer.seq_len
device = VAE_LSTM_dropout_one_layer.device

'''
dropout_rate = VAE_LSTM_dropout.dropout_rate
embedding_dim = VAE_LSTM_dropout.embedding_dim
learning_rate = VAE_LSTM_dropout.learning_rate
batch_size = VAE_LSTM_dropout.batch_size
seq_len = VAE_LSTM_dropout.seq_len
device = VAE_LSTM_dropout.device
'''

# Define the dataset class
class TimeSeriesDataset(Dataset):
    def __init__(self, data, mask, seq_len):
        self.data = data
        self.mask = mask
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len + 1

    def __getitem__(self, idx):
        data_seq = self.data[idx:idx+self.seq_len]
        mask_seq = self.mask[idx:idx+self.seq_len]
        return data_seq, mask_seq

# Load data
data_missing = pd.read_csv("data_missing_10_percent.csv")
data_complete = pd.read_csv("data_missing_0_percent.csv")

# Drop first column due to csv creation
data_missing = data_missing.iloc[: , 1:]

# Copy dataset in order to prevent altering (will be used later on)
df_original = data_missing.copy()

# Manually split data into 70% for training, 10% for validation, and 20% for imputation
train_size = int(0.7 * len(data_missing))
val_size = int(0.1 * len(data_missing))
impute_size = len(data_missing) - train_size - val_size

# Split data into train, validation, and test
train_data = data_missing[:train_size]
val_data = data_missing[train_size:train_size + val_size]
impute_data = data_missing[train_size + val_size:]

testing_data = data_complete[train_size + val_size:]

#count the total number of missing values of data to be imputed
impute_missing = impute_data.isnull().sum().sum()

# Save data to be imputed
#impute_data.to_csv('data_to_be_imputed.csv', index=False)

# Create mask for missing values
mask = data_missing.notna().astype(int)

# Impute and scale the data
imputer = SimpleImputer(strategy='mean')
min_max_scaler = MinMaxScaler()

# Impute values using "mean" strategy and scale data
train_data = pd.DataFrame(imputer.fit_transform(train_data), columns=train_data.columns)
train_data_scaled = min_max_scaler.fit_transform(train_data)
mask_train = mask[:train_size].values

# Using transform function not to "leak" information to val dataset 
val_data = pd.DataFrame(imputer.transform(val_data), columns=val_data.columns)
val_data_scaled = min_max_scaler.transform(val_data)
mask_val = mask[train_size:train_size + val_size].values

# Using transform function not to "leak" information to test dataset 
impute_data = pd.DataFrame(imputer.transform(impute_data), columns=impute_data.columns)
impute_data_scaled = min_max_scaler.transform(impute_data)
mask_impute = mask[train_size + val_size:].values

# Using transform function not to "leak" information to test dataset 
testing_data_scaled = min_max_scaler.transform(testing_data)
testing_data_scaled_df = pd.DataFrame(testing_data_scaled, columns=impute_data.columns)

# Creation of dataloaders (shuffle=False since ordering matters)
train_dataset = TimeSeriesDataset(train_data_scaled, mask_train, seq_len)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

val_dataset = TimeSeriesDataset(val_data_scaled, mask_val, seq_len)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

impute_dataset = TimeSeriesDataset(impute_data_scaled, mask_impute, seq_len)
impute_dataloader = DataLoader(impute_dataset, batch_size=1, shuffle=False)

# Initialize model, loss function and optimizer
model = RecurrentVAE(
    seq_len, 
    train_data_scaled.shape[1], 
    embedding_dim=embedding_dim,
    dropout_rate=dropout_rate,
    device=device
)

'''
model = RecurrentVAE_one_layer(
    seq_len, 
    train_data_scaled.shape[1], 
    embedding_dim=embedding_dim,
    dropout_rate=dropout_rate,
    device=device
)
'''

criterion = nn.MSELoss(reduction='none')
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Function to compute loss for VAE
def vae_loss(recon_x, x, mean, log_var, mask):
    BCE = criterion(recon_x, x)
    BCE = (BCE * mask).sum() / mask.sum()
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return BCE + KLD / x.shape[0]

# Training loop with tqdm progress bar
# Prepare the plot
plt.ion()
fig, ax = plt.subplots()
train_losses = []
val_losses = []
# Initialize variables for early stopping
best_val_loss = float('inf')
epochs_with_no_improve = 0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    with tqdm(train_dataloader, unit="batch") as tepoch:
        for batch in tepoch:
            tepoch.set_description(f"Epoch {epoch+1}/{num_epochs}")

            data_seq, mask_seq = batch
            data_seq = data_seq.float().to(device)
            mask_seq = mask_seq.float().to(device)
            
            output, mean, log_var = model(data_seq)
            loss = vae_loss(output, data_seq, mean, log_var, mask_seq)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            tepoch.set_postfix(loss=running_loss/len(train_dataloader))
    epoch_train_loss = running_loss / len(train_dataloader)
    train_losses.append(epoch_train_loss)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_dataloader):.4f}')

    # Validation phase
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_dataloader:
            data_seq, mask_seq = batch
            data_seq = data_seq.float().to(device)
            mask_seq = mask_seq.float().to(device)
            
            output, mean, log_var = model(data_seq)
            loss = vae_loss(output, data_seq, mean, log_var, mask_seq)
            val_loss += loss.item()
    
    val_loss /= len(val_dataloader)
    val_losses.append(val_loss)
    print(f'Validation Loss: {val_loss:.4f}')

    # Check for early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_with_no_improve = 0
    else:
        epochs_with_no_improve += 1
    
    if epochs_with_no_improve >= patience and epoch >= 0.6*num_epochs:
        print(f'Early stopping criteria raeched, at epoch: {epoch}')
        break
    print(f'Current number of epochs with no improve from {round(best_val_loss,4)} is:{epochs_with_no_improve}')
    # Update the plot

    # Update the plot
    ax.clear()
    ax.plot(train_losses, label='Training Loss')
    ax.plot(val_losses, label='Validation Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss Over Time')
    ax.legend()
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.01)

plt.ioff()
plt.show()

print("Training complete.")

# Imputing function
model.eval()
imputed_values = []

with torch.no_grad():
    for batch in impute_dataloader:
        data_seq, mask_seq = batch
        data_seq = data_seq.float().to(device)

        output, mean, log_var = model(data_seq)  # Unpack the returned tuple
        imputed_values.append(output.squeeze(0).cpu().numpy())

imputed_values = np.concatenate(imputed_values, axis=0)
imputed_values = imputed_values[:len(impute_data_scaled)]

impute_df_original = df_original.iloc[train_size + val_size:train_size + val_size + len(imputed_values)]
impute_df = pd.DataFrame(imputed_values, columns=df_original.columns, index=impute_df_original.index)

impute_data_combined = impute_data_scaled.copy()
impute_data_combined[mask_impute == 0] = impute_df.values[mask_impute == 0]

imputed_df_scaled = pd.DataFrame(impute_data_combined, columns=df_original.columns)

impute_df_unscaled = pd.DataFrame(min_max_scaler.inverse_transform(impute_data_combined), columns=df_original.columns, index=impute_df_original.index)
#impute_df_unscaled.to_csv('imputed_data_VAE.csv', index=False)
#print("Imputed data saved to imputed_data.csv.")

def check_ks_test(method_name):
    file_path = os.path.join(os.getcwd(), "p-values.txt")
    print("starting the check_ks_test function")
    pvalues=[]
    for column in df_original.columns:
        pvalue = f"pvalue of column {column} is: {(stats.ks_2samp(method_name[column], testing_data_scaled_df[column]))[1]}"
        pvalues.append(pvalue)
    with open(file_path, "w") as file:
        for item in pvalues:
            file.write("%s\n" % item)
    print(f"The p values of the iterative imputed dataframe are: {pvalues}")
#check_ks_test(imputed_df_scaled)

def find_RMSE(method):
    temp1 = testing_data_scaled_df.sub(method)
    temp2 = temp1.mul(temp1)
    RMSE = math.sqrt((temp2.sum().sum())/impute_missing)
    print(f" The RMSE value of the iterative imputed dataframe is: {RMSE}")
find_RMSE(imputed_df_scaled)


