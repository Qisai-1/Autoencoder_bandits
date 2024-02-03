import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import numpy as np
import itertools


from sklearn.datasets import load_digits
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt
import argparse


# Argument parser setup
parser = argparse.ArgumentParser(description='Autoencoder Configuration')
parser.add_argument('--model', type=str, choices=['1', '2', '3'], default='3', help='Model to use for inference')
parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda' if torch.cuda.is_available() else 'cpu', help='Computation device')
args = parser.parse_args()


# Model configuration dictionary
model_configs = {
    '1': {
        'module': 'model_1',
        'model_path': 'models/model_1/Model_1_latent_addon_onehot.pth'
    },
    '2': {
        'module': 'model_2',
        'model_path': 'models/model_2/Model_2_input_env_onehot_reconstruct_env.pth'
    },
    '3': {
        'module': 'model_3',
        'model_path': 'models/model_3/Model_3_input_env_onehot.pth'
    }
}

# Dynamic import
def import_models(module_name):
    Autoencoder = getattr(__import__(module_name, fromlist=['Autoencoder']), 'Autoencoder')
    Linear_Mapping = getattr(__import__(module_name, fromlist=['Linear_Mapping']), 'Linear_Mapping')
    Encoder_soil_features = getattr(__import__(module_name, fromlist=['Encoder_soil_features']), 'Encoder_soil_features')
    Decoder_soil_features = getattr(__import__(module_name, fromlist=['Decoder_soil_features']), 'Decoder_soil_features')
    return Autoencoder, Linear_Mapping, Encoder_soil_features, Decoder_soil_features

# Select and load model based on command-line argument
config = model_configs[args.model]
Autoencoder, Linear_Mapping, Encoder_soil_features, Decoder_soil_features = import_models(config['module'])
model_path = config['model_path']

if args.model == '1':
    loss_weight = 0.65
    latent_dim = 26
    # feature_size = 34
    # decoder_output_size = 42
    hidden_size = 34
    
if args.model == '2':
    loss_weight = 0.65
    latent_dim = 18
    # feature_size = 42
    # decoder_output_size = 34
    hidden_size = 18
    
if args.model == '3':
    loss_weight = 0.65
    latent_dim = 26
    # feature_size = 34
    # decoder_output_size = 34
    hidden_size = 34
    
    
    
# # Initialize model instances
# autoencoder = Autoencoder(latent_dim,feature_size,decoder_output_size)
# linear_model = Linear_Mapping(hidden_size)

device = torch.device(args.device)

# Loading both models
saved_states = torch.load(model_path, map_location=device)

# autoencoder.load_state_dict(saved_states['autoencoder'])
# linear_model.load_state_dict(saved_states['linear_model'])
autoencoder = saved_states['autoencoder']
linear_model = saved_states['linear_model']


autoencoder.to(device)
linear_model.to(device)


from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler

class YieldDataset(Dataset):
    def __init__(self, csv_file, exclude_cols=['cultivar'], target_col='yield_kg_ha', normalize=True):
        # Load data
        self.df = pd.read_csv(csv_file)
        
        # Extract seed names and remove specified columns
        self.seed_id = self.df['cultivar'].unique().tolist()
        self.data = self.df.loc[:, ~self.df.columns.isin(exclude_cols)]
        
        # One-hot encode the 'cultivar' column
        encoder = OneHotEncoder(sparse=False)
        self.seed_one_hot = encoder.fit_transform(self.df[['cultivar']])
        
        # Normalize other features if specified
        if normalize:
            scaler = StandardScaler()
            self.data = scaler.fit_transform(self.data.to_numpy())
        else:
            self.data = self.data.to_numpy()
        
        # Combine one-hot encoded cultivar with other features
        self.obs = np.hstack((self.seed_one_hot, self.data))
        
        # Extract target values
        self.targets = self.df[target_col].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.obs[idx], self.targets[idx]
     
# Initialize your dataset
dataset = YieldDataset(csv_file="dataset/clean_yield.csv", normalize=True)

# Prepare DataLoader for evaluation
test_dataloader = DataLoader(dataset, batch_size=64, shuffle=False)   
    

# Don't forget to set them to evaluation mode if you're doing inference
autoencoder.eval()
linear_model.eval()
criterion = nn.MSELoss()
unique_seed = len(dataset.seed_id)

test_loss = 0.0
test_prediction = []
yield_true = []
latent =[]
with torch.no_grad():

    for test_batch  in test_dataloader:
        test_batch = test_batch[0].to(torch.float32).to(device)
        true_yield = test_batch[:,-1:]
        inputs_soil = test_batch[:,unique_seed:-1]
        inputs_onehot = test_batch[:, :unique_seed]
        
        if args.model == '1':

            latent_out, soil_out = autoencoder(inputs_onehot, inputs_soil)

            combined_soil_onehot = torch.cat((inputs_soil, inputs_onehot), dim=1)

            loss_combine_auto = criterion(combined_soil_onehot, soil_out)
            yield_result_test = linear_model(latent_out)
            
        if  args.model == '2':
            auto_input = torch.cat((inputs_onehot, inputs_soil), dim=1)
            # Forward passinputs_soilnputs_onehot, inputs_soil)
            latent_out, soil_out = autoencoder(auto_input)
            
            loss_combine_auto = criterion(soil_out, inputs_soil)
            yield_result_test = linear_model(latent_out)
            
        if  args.model == '3':
            latent_out, soil_out = autoencoder(inputs_soil)
            
            loss_combine_auto = criterion(soil_out, inputs_soil)
            
            combined_soil_onehot = torch.cat((latent_out, inputs_onehot), dim=1)
            yield_result_test = linear_model(combined_soil_onehot)
            
            latent_out = combined_soil_onehot
            
            
        latent.extend(latent_out.cpu().numpy())
        test_prediction.extend(yield_result_test.cpu().numpy())
        yield_true.extend(true_yield.cpu().numpy())

        loss_liner_map = criterion(true_yield, yield_result_test)
        loss = (1-loss_weight) * loss_combine_auto + loss_weight * loss_liner_map
        test_loss += loss.item()

average_test_loss = test_loss / len(test_dataloader)
print(f"Average Test Loss: {average_test_loss:.4f}")

# Assuming test_prediction and yield_true are lists of numbers or single numpy arrays
test_prediction_array = np.array(test_prediction).reshape(-1, 1)  # Ensure it's a 2D column vector
yield_true_array = np.array(yield_true).reshape(-1, 1)  # Ensure it's a 2D column vector
latent_array = np.array(latent).reshape(-1, hidden_size)  # Ensure it's a 2D array, adjust latent_dim as necessary


results_array = np.hstack((latent_array, test_prediction_array, yield_true_array))


# Create dynamic header labels for latent features
latent_headers = [f"Latent_{i+1}" for i in range(latent_dim)]

# Combine headers for latent features, predicted yield, and true yield
headers = ",".join(latent_headers + ["Predicted_Yield", "True_Yield"])

# Save the array to CSV, including the new header
np.savetxt(f"model_result/modl_{args.model}_results_with_latent.csv", results_array, delimiter=",", header=headers, comments='', fmt='%f')