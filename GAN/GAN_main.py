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
from GAN import Generator, Discriminator
import GAN
import matplotlib.pyplot as plt
import math
from scipy import stats

num_epochs = 300
patience = 20
weight_decay = 0
n_features = 66

#choose parameters specific to model

dropout_rate = GAN.dropout_rate
hidden_dim = GAN.hidden_dim
learning_rate = GAN.learning_rate
batch_size = GAN.batch_size
seq_len = GAN.seq_len
device = GAN.device

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
data_missing = data_missing.iloc[:, 1:]

# Copy dataset to prevent altering (will be used later)
df_original = data_missing.copy()

# Manually split data into 70% for training, 10% for validation, and 20% for imputation
train_size = int(0.7 * len(data_missing))
val_size = int(0.1 * len(data_missing))
impute_size = len(data_missing) - train_size - val_size

# Split data into train, validation, and imputation sets
train_data = data_missing[:train_size]
val_data = data_missing[train_size:train_size + val_size]
impute_data = data_missing[train_size + val_size:]

testing_data = data_complete[train_size + val_size:]

#count the total number of missing values of data to be imputed
impute_missing = impute_data.isnull().sum().sum()

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

# Using transform function not to "leak" information to impute dataset 
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

# Define how many times to train the generator for each discriminator update
generator_updates_per_discriminator_update = 5

# Initialize models
generator = Generator(seq_len, n_features).to(device)
discriminator = Discriminator(seq_len, n_features).to(device)

# Loss and optimizers
criterion = nn.BCELoss()
optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate)
optimizer_d = optim.Adam(discriminator.parameters(), lr=learning_rate)

# Initialize variables for early stopping
best_val_loss = float('inf')
epochs_with_no_improve = 0

# Training loop with tqdm and validation phase
for epoch in tqdm(range(num_epochs), desc="Training Progress"):
    generator.train()  # Ensure the generator is in training mode
    epoch_loss_d_real = 0
    epoch_loss_d_fake = 0
    epoch_loss_g = 0

    # Training phase
    for real_data, real_mask in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
        real_data = real_data.to(device).float()  # Convert to float32
        real_mask = real_mask.to(device).float()

        # Train Discriminator
        optimizer_d.zero_grad()
        
        # Create labels for real and fake data
        batch_size = real_data.size(0)
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # Discriminator on real data
        outputs = discriminator(real_data)
        loss_d_real = criterion(outputs, real_labels)
        loss_d_real.backward()
        epoch_loss_d_real += loss_d_real.item()

        # Generator attempts to impute missing data
        fake_data = generator(real_data)
        fake_data[real_mask == 1] = real_data[real_mask == 1]  # Preserve real values where they exist
        outputs = discriminator(fake_data.detach())
        loss_d_fake = criterion(outputs, fake_labels)
        loss_d_fake.backward()
        optimizer_d.step()
        epoch_loss_d_fake += loss_d_fake.item()

        # Train Generator multiple times
        for _ in range(generator_updates_per_discriminator_update):
            optimizer_g.zero_grad()
            
            # Generator forward pass
            fake_data = generator(real_data)
            fake_data[real_mask == 1] = real_data[real_mask == 1]  # Preserve real values where they exist
            outputs = discriminator(fake_data)
            loss_g = criterion(outputs, real_labels)  # We want the generator to fool the discriminator
            loss_g.backward()
            optimizer_g.step()
            epoch_loss_g += loss_g.item()

    tqdm.write(f'Epoch [{epoch+1}/{num_epochs}], Loss D: {epoch_loss_d_real + epoch_loss_d_fake:.4f}, Loss G: {epoch_loss_g:.4f}')
    
    # Validation phase
    generator.eval()  # Switch generator to evaluation mode
    val_loss = 0.0
    validation_criterion = nn.MSELoss()  # Use MSELoss for validation
    with torch.no_grad():
        for data_seq, mask_seq in tqdm(val_dataloader, desc=f"Validation {epoch+1}/{num_epochs}", leave=False):
            data_seq = data_seq.float().to(device)
            mask_seq = mask_seq.float().to(device)
            
            output = generator(data_seq)
            loss = validation_criterion(output, data_seq)
            masked_loss = (loss * mask_seq).sum() / mask_seq.sum()
            val_loss += masked_loss.item()
    
    val_loss /= len(val_dataloader)
    tqdm.write(f'Validation Loss: {val_loss:.4f}')

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_with_no_improve = 0
    else:
        epochs_with_no_improve += 1
    
    if epochs_with_no_improve >= patience and epoch >= 0.6*num_epochs:
        print(f'Early stopping criteria raeched, at epoch: {epoch}')
        break
    print(f'Current number of epochs with no improve from {round(best_val_loss,4)} is:{epochs_with_no_improve}')

    generator.train()  # Ensure the generator is back to training mode after validation

# Save the generator model if needed
#torch.save(generator.state_dict(), 'generator.pth')

# Use trained model to impute data
imputed_values = []

with torch.no_grad():
    for batch in impute_dataloader:
        data_seq, mask_seq = batch
        data_seq = data_seq.float().to(device)
        
        output = generator(data_seq)
        imputed_values.append(output.squeeze(0).cpu().numpy())

# Convert imputed_values to a DataFrame
imputed_values = np.concatenate(imputed_values, axis=0)
imputed_values = imputed_values[:len(impute_data_scaled)]  # Ensure the correct length

# Construct the imputed dataframe for the 20% impute_data
impute_df_original = df_original.iloc[train_size + val_size:train_size + val_size + len(imputed_values)]
impute_df = pd.DataFrame(imputed_values, columns=df_original.columns, index=impute_df_original.index)

# Combine imputed values with the original impute_data to leave observed values untouched
impute_data_combined = impute_data_scaled.copy()
impute_data_combined[mask_impute == 0] = impute_df.values[mask_impute == 0]

imputed_df_scaled = pd.DataFrame(impute_data_combined, columns=df_original.columns)

# Unscale imputed dataset back to original numbers
impute_df_unscaled = pd.DataFrame(min_max_scaler.inverse_transform(impute_data_combined), columns=df_original.columns, index=impute_df_original.index)

# Save the imputed dataframe to CSV
#impute_df_unscaled.to_csv('imputed_data_GAN.csv', index=False)
#print("Imputed data saved to imputed_data_GAN.csv.")


def find_RMSE(method):
    temp1 = testing_data_scaled_df.sub(method)
    temp2 = temp1.mul(temp1)
    RMSE = math.sqrt((temp2.sum().sum())/impute_missing)
    print(f" The RMSE value of the iterative imputed dataframe is: {RMSE}")
find_RMSE(imputed_df_scaled)