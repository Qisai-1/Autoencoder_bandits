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

# from sklearn.decomposition import PCA
# pca_30 = PCA(n_components=30)
# pca_result_30 = pca_30.fit_transform(obs)
# print('Cumulative explained variation for 50 principal components: {}'.format(np.sum(pca_30.explained_variance_ratio_)))


# normalize dataset
def normalize_data(data):
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(data)
    return X_normalized

# Define the autoencoder architectrue
class Encoder_one_hot(nn.Module):
    def __init__(self,
                num_input_size : int,  
                latent_dim : int):
        super().__init__()

        self.fc1 = nn.Linear(num_input_size, out_featrues=30)
        #self.fc2 = nn.Linear(in_features=18, out_features=16)
        self.fc3 = nn.Linear(in_features=30, out_features=latent_dim)

    def forward(self, x):

        x = torch.relu(self.fc1(x))
        #x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return x

# class Decoder_one_hot(nn.Module):

#     def __init__(self,
#                 num_input_size : int,
#                 latent_dim : int):
#         super().__init__()

#         self.fc1 = nn.Linear(latent_dim, out_features=6)
#         #self.fc2 = nn.Linear(in_features=16, out_features=16)
#         self.fc3 = nn.Linear(in_features=6, out_features=num_input_size)

#     def forward(self, x):

#         x = torch.relu(self.fc1(x))
#         #x = torch.relu(self.fc2(x))
#         x = self.fc3(x)
#         x = nn.Identity()(x) # Using Tanh activation in the decoder
#         return x


def plot_loss_subplots(train_losses_list, val_losses_list,titles_list):

    num_plots = len(train_losses_list)

    # Create subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    for i in range(num_plots):
        row = i // 2
        col = i % 2

        epochs = np.arange(1, len(train_losses_list[i]) + 1)
        axs[row, col].plot(epochs, train_losses_list[i], label='Train Loss')
        axs[row, col].plot(epochs, val_losses_list[i], label='Validation Loss')
        axs[row, col].set_title(titles_list[i])
        axs[row, col].set_xlabel('Epoch')
        axs[row, col].set_ylabel('Loss')
        axs[row, col].legend()

    # Adjust layout
    plt.tight_layout()

    # Show the plots
    #plt.show()

    # Close the plot window automatically after 5 seconds

    #plt.close()
    plt.savefig('plots/model_2.png')

def scatter_plot(ax, prediction, true_labels, xlabel='True Labels', ylabel='Predicted Values', title='Scatter Plot'):
    ax.scatter(true_labels, prediction, label='Scatter Plot')
    ax.plot([min(true_labels), max(true_labels)], [min(true_labels), max(true_labels)], linestyle='--', color='red', label='45-degree line')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()


class Encoder_soil_features(nn.Module):

    def __init__(self,
                num_input_size : int,  
                latent_dim : int):
        super().__init__()

        self.fc1 = nn.Linear(num_input_size, out_features=30)
        #self.fc2 = nn.Linear(in_features=16, out_features=10)
        self.fc3 = nn.Linear(in_features=30, out_features=latent_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        #x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return x


class Decoder_soil_features(nn.Module):

    def __init__(self,
                num_output_size : int,
                latent_dim : int):
        super().__init__()

        self.fc1 = nn.Linear(latent_dim, out_features=30)
        #self.fc2 = nn.Linear(in_features=16, out_features=10)
        self.fc3 = nn.Linear(in_features=30, out_features=num_output_size)

    def forward(self, x):
 
        x = torch.relu(self.fc1(x))
        #x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        x = nn.Tanh()(x)
        return x

 

class Autoencoder(nn.Module):

    def __init__(self,
                 latent_dim_soil_features: int,
                 num_input_features : int,
                 decoder_output_size : int,
                 encoder_soil_features : object = Encoder_soil_features,
                 decoder_soil_features : object = Decoder_soil_features,
                 ):
        super(Autoencoder, self).__init__()
        self.encoder_soil = encoder_soil_features(num_input_features,latent_dim_soil_features)
        self.decoder_soil = decoder_soil_features(decoder_output_size,latent_dim_soil_features)

    def forward(self, soil):
        soil_latent = self.encoder_soil(soil)

        soil = self.decoder_soil(soil_latent)

        return  soil_latent, soil


def Dataset_split(obs,batch_size):

    # Split dataset into training, validation, and testing sets
    train_data, test_data = train_test_split(obs, test_size=0.2, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)

    # Create DataLoaders for training, validation, and testing
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)


    return train_dataloader, val_dataloader, test_dataloader,  train_data, val_data , test_data
 
# Define the architecture of the DNN for linear mapping 
class Linear_Mapping(nn.Module):
    def __init__(self,
                latent_dim : int):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, 1)  # Input layer

    def forward(self, x):
        x = self.fc1(x)  # No activation in input layer make linear relationship

        return x





if __name__ == "__main__":
    import argparse

    # Argument parser setup
    parser = argparse.ArgumentParser(description='Autoencoder Configuration')
    parser.add_argument('--latent_size', type=int, default=18, help='Size of the latent feature')
    parser.add_argument('--first_train_phase', type=int, default=50, help='Number of epochs for the first training phase')
    parser.add_argument('--loss_weight', type=float, default=0.65, help='Weight for the linear mapping loss')
    parser.add_argument('--total_epochs', type=int, default=200, help='Total number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--eval_mode', type=bool, default=False, help='Evaluation mode')
    args = parser.parse_args()



    # Set the random seed for Python's built-in random number generator
    seed = 42
    np.random.seed(seed)

    # Set the random seed for PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # If you are using CUDA

    # load subset of data used to train bilinear model
    subset_df = pd.read_csv("dataset/clean_yield.csv")

    # extract field names, seed names and field x seed name combinations
    seed_id = subset_df['cultivar'].unique().tolist()
    print('Example of seed :', len(seed_id))

    # Get a list of column names to exclude
    exclude_cols = ['cultivar'] # ,'yield_kg_ha'

    
    # Build a sub-DataFrame with all columns except the excluded one
    sub_data = subset_df.loc[:, ~subset_df.columns.isin(exclude_cols)]

    # Create the OneHotEncoder object
    encoder = OneHotEncoder()

    # one hot encoding the Pedigree
    one_hot = encoder.fit_transform(np.array(subset_df["cultivar"]).reshape(-1, 1))
    seed_one_hot = one_hot.toarray()
    obs_raw = sub_data.to_numpy()


    # # check unique one hot enconding

    # unique_lists = np.unique(seed_one_hot.view(np.dtype((np.void, seed_one_hot.dtype.itemsize * seed_one_hot.shape[1])))).view(seed_one_hot.dtype).reshape(-1, seed_one_hot.shape[1])
    true_yield = subset_df['yield_kg_ha'].values

    # Combine one hot (cultivar) with soil feature value
    obs = []

    for i in range(len(obs_raw)):
        obs.append(list(seed_one_hot[i]) + list(obs_raw[i]))
    obs = np.asarray(obs)

    # Normalize the imputed dataset

    unique_seed = len(seed_one_hot[0])
    obs[:,unique_seed:] = normalize_data(obs[:,unique_seed:])

    # Set device to use GPU if available, else use CPU
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print("Device:", device)

    # Initialize autoencoder

    feature_size = obs.shape[1] -1 # 1 is the yield column
    decoder_output_size = obs.shape[1] - unique_seed -1

    latent_dim = args.latent_size
    first_train_phase = args.first_train_phase
    loss_weight = args.loss_weight
    num_epochs = args.total_epochs
    batch_size = args.batch_size
    eval_mode = args.eval_mode
    
    autoencoder = Autoencoder(latent_dim, feature_size, decoder_output_size ).to(device)
    train_dataloader, val_dataloader, test_dataloader,  train_data, val_data , test_data = Dataset_split(obs, batch_size)
    criterion = nn.MSELoss()

    # Initialize the model, loss function, and optimizer 
    mapping = Linear_Mapping(latent_dim).to(device)

    # Separate encoder, decoder, and DNN parameters

    #encoder_params_combine = list(autoencoder.encoder_onehot.parameters())
    encoder_params_soil = list(autoencoder.encoder_soil.parameters())
    decoder_params_soil = list(autoencoder.decoder_soil.parameters())
    dnn_params = list(mapping.parameters())



    # Create an optimizer
    optimizer = optim.SGD([
        {'params': encoder_params_soil, 'lr': 0.000},
        {'params': decoder_params_soil, 'lr': 0.000},
        {'params': dnn_params, 'lr': 0.0000}
    ])

    # mapping = Linear_Mapping(latent_dim).to(device)
    # optimizer_auto = optim.Adam(autoencoder.parameters(), lr=0.003)
    # parameters = list(autoencoder.parameters()) + list(mapping.parameters())
    # optimizer_combine = optim.Adam(parameters, lr=0.003)


    train_losses = []
    train_losses_linear = []
    train_losses_auto = []

    val_losses = []
    val_losses_linear = []
    val_losses_auto = []

    for epoch in range(num_epochs):

        running_loss = 0.0
        running_loss_auto = 0.0
        running_loss_linear = 0.0

        for batch_train in train_dataloader:
            optimizer.zero_grad()
            input_batch = batch_train.to(torch.float32).to(device)
            true_yield = input_batch[:,-1:]
            inputs_soil = input_batch[:,unique_seed:-1]
            inputs_onehot = input_batch[:, :unique_seed]
 
            # auto_input = torch.cat((inputs_onehot, inputs_soil), dim=1)
            # Forward passinputs_soilnputs_onehot, inputs_soil)
            latent_out, soil_out = autoencoder(input_batch[:,:-1])
            
            if epoch < first_train_phase:
                # Only update specific encoder and decoder parameters
                optimizer.param_groups[0]['lr'] = 0.003
                optimizer.param_groups[1]['lr'] = 0.003
                optimizer.param_groups[2]['lr'] = 0


                loss_soil = criterion(soil_out, inputs_soil)
                loss_soil.backward()
                optimizer.step()
                running_loss += loss_soil.item()

            else:
                
                 # Only update specific encoder and decoder parameters, and linear model
                optimizer.param_groups[0]['lr'] = 0.003
                optimizer.param_groups[1]['lr'] = 0.003
                optimizer.param_groups[2]['lr'] = 0.003

                #combined_soil_onehot = torch.cat((latent_out, inputs_onehot), dim=1)
                yield_result = mapping(latent_out)
                loss_liner_map = criterion(true_yield, yield_result)
                loss_soil = criterion(soil_out, inputs_soil)
    
                # Backpropagation and optimization
                loss = loss_weight * loss_liner_map + (1 - loss_weight) * loss_soil
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                running_loss_auto += loss_soil.item()
                running_loss_linear += loss_liner_map.item()

        # Total Training Loss
        train_loss = running_loss / len(train_dataloader)
        train_losses.append(train_loss)

        if epoch >= first_train_phase:
            # Training Linear Loss 
            train_loss_auto = running_loss_auto / len(train_dataloader)
            train_losses_auto.append(train_loss_auto)

            # Training Auto Loss 
            train_loss_linear = running_loss_linear / len(train_dataloader)
            train_losses_linear.append(train_loss_linear)

        autoencoder.eval()
        val_loss = 0.0
        val_loss_auto = 0.0
        val_loss_linear = 0.0

        with torch.no_grad():

            for batch_val in val_dataloader:

                val_batch = batch_val.to(torch.float32).to(device)
                true_yield = val_batch[:,-1:]
                inputs_soil = val_batch[:,unique_seed:-1]
                inputs_onehot = val_batch[:, :unique_seed]

                latent_out, soil_out = autoencoder(val_batch[:,:-1])
                #loss_onehot = criterion(onehot_out, inputs_onehot)
                
                if epoch < first_train_phase:
                    loss = criterion(soil_out, inputs_soil)
                    val_loss += loss.item()

                else:
                    yield_result = mapping(latent_out)
                    loss_liner_map = criterion(true_yield, yield_result)
                    loss_soil = criterion(soil_out, inputs_soil)


                    loss = loss_weight * loss_liner_map + (1 - loss_weight) * loss_soil
                    val_loss += loss.item()
                    val_loss_auto += loss_soil.item()
                    val_loss_linear += loss_liner_map.item()

        # Total val Loss
        average_val_loss = val_loss / len(val_dataloader)
        val_losses.append(average_val_loss)
    
        if epoch >= first_train_phase:
            # Val Linear Loss 
            average_val_loss_auto = val_loss_auto / len(val_dataloader)
            val_losses_auto.append(average_val_loss_auto)

            # Val Auto Loss 
            average_val_loss_linear = val_loss_linear / len(val_dataloader)
            val_losses_linear.append(average_val_loss_linear)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, Validation Loss: {average_val_loss:.4f}")


    # Create subplots
    train_loss_list = [train_losses,train_losses_linear,train_losses_auto,train_losses[:first_train_phase]]
    val_loss_list = [val_losses,val_losses_linear,val_losses_auto,val_losses[:first_train_phase]]
    titles = ['Total loss', 'Linear loss', 'Autoencoder loss', f'First{first_train_phase}epochs Autoencoder loss']
    plot_loss_subplots(train_loss_list, val_loss_list, titles)


    # Calculate R-squared
    val_yield = torch.tensor(val_data[:,-1], dtype=torch.float32).to(device)
    val_total_data = torch.tensor(val_data[:,:-1], dtype=torch.float32).to(device)
    latent_out, soil_out = autoencoder(val_total_data)
    yield_pred = mapping(latent_out)
    
    r_squared = r2_score(val_data[:,-1], yield_pred.cpu().detach().numpy())
    print(f'R-squared value: {r_squared:.4f}')

    # Testing loop
    autoencoder.eval()
    mapping.eval()
    test_loss = 0.0
    test_prediction = []
    yield_true = []
    with torch.no_grad():

        for test_batch  in test_dataloader:
            test_batch = test_batch.to(torch.float32).to(device)
            true_yield = test_batch[:,-1:]
            inputs_soil = test_batch[:,unique_seed:-1]
            inputs_onehot = test_batch[:, :unique_seed]

            auto_input = torch.cat((inputs_onehot, inputs_soil), dim=1)
            # Forward passinputs_soilnputs_onehot, inputs_soil)
            latent_out, soil_out = autoencoder(auto_input)
            
            loss_combine_auto = criterion(soil_out, inputs_soil)
            yield_result_test = mapping(latent_out)

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

    # Stack the two arrays horizontally to create a two-column array
    results_array = np.hstack((test_prediction_array, yield_true_array))

    # Save the results to a CSV file
    if eval_mode:
        np.savetxt("test_results.csv", results_array, delimiter=",", header="Predicted_Yield,True_Yield", comments='', fmt='%f')

    # Create a scatter plot

    fig, ax = plt.subplots()
    scatter_plot(ax, test_prediction, yield_true)

    # Save the plot to a file (e.g., PNG, PDF, SVG)
    plt.savefig('plots/model_2_scatter.png')
    torch.save({
        'autoencoder': autoencoder,
        'linear_model': mapping
    }, 'models/model_2/Model_2_input_env_onehot_reconstruct_env.pth')
        
    # Show the plot
    # plt.show()
