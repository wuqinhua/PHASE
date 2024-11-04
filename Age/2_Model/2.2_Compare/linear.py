import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score, mean_squared_error
from tqdm import tqdm
from scipy.stats import pearsonr
import pickle

os.chdir("/data/wuqinhua/phase/age/")

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load datasets
with open('./age_datalist_20240908.pickle', 'rb') as handle:
    DataList = pickle.load(handle)

with open('./age_datalabel_20240908.pickle', 'rb') as handle:
    DataLabel = pickle.load(handle)

def seed_torch(seed=777):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# Build Pytorch Dataset
class COVID_Dataset(Dataset):
    def __init__(self, DataList, DataLabel):
        super().__init__()
        self.DataList = DataList
        self.DataLabel = DataLabel

    def __len__(self):
        return self.DataLabel.shape[0]

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        dataList = self.DataList[index].to(device)
        dataLabel = self.DataLabel[index].to(device)
        return dataList, dataLabel

# Define Linear Regression model
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        out = self.linear(x)
        return out

class LinearRegressionModel(nn.Module):
    def __init__(self, dim, states=1):  
        super(LinearRegressionModel, self).__init__()
        self.states = states
        if states == 1:
            self.layer = torch.nn.Linear(dim, states).to(device)
        else:
            self.layer = torch.nn.Linear(dim, states - 1).to(device)
        
        self.layer.weight.data.zero_()
        if self.layer.bias is not None:
            self.layer.bias.data.zero_()

    def forward(self, x):
        # If states == 1, sum over the second-to-last dimension
        if self.states == 1:
            return torch.sum(self.layer(x).to(device), dim=-2, keepdim=True)
        
        # Otherwise, concatenate a zero tensor and the summed output
        else:
            return torch.cat([torch.zeros(1, 1).to(device), torch.sum(self.layer(x).to(device), dim=-2, keepdim=True)], dim=1)

seed_torch(1)

# Cross Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
list_r2 = []
list_rmse = []
list_cor = []
lr = 0.0000001

for fold, (train_idx, valid_idx) in enumerate(skf.split(DataList, DataLabel)):
    print(f'FOLD {fold}')
    print('--------------------------------')
    
    # Prepare train and validation data
    train_data = [DataList[i] for i in train_idx]
    train_label = DataLabel[train_idx]
    valid_data = [DataList[i] for i in valid_idx]
    valid_label = DataLabel[valid_idx]
    
    # Create the linear regression model
    network = LinearRegressionModel(dim=5000, states=1).to(device)
    trainData = COVID_Dataset(train_data, train_label)
    validData = COVID_Dataset(valid_data, valid_label)
    trainLoader = DataLoader(trainData, batch_size=1)
    validLoader = DataLoader(validData, batch_size=1)

    # Define loss and optimizer
    criterion = nn.MSELoss()  # Mean Squared Error for regression
    optimizer = optim.Adam(network.parameters(), lr=lr)

    # Train the model
    epochs = 1000
    max_r2 = float("-inf")
    
    for epoch in range(epochs):
        progress_bar = tqdm(trainLoader, desc=f'Epoch {epoch + 1}/{epochs}', leave=False)
        running_loss = 0.0
        network.train()
        
        for inputs, labels in progress_bar:
            optimizer.zero_grad()
            outputs = network(inputs[0])
            labels = labels.view(-1, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            progress_bar.set_postfix(loss=running_loss)
            
        avg_loss = running_loss / len(trainLoader)

        # Validation
        network.eval()
        y_true = []
        y_pred = []
        
        for inputs, labels in validLoader:
            # print(f"Inputs shape: {inputs[0].shape}, Labels shape: {labels.shape}")
            labels = labels.float().view(-1, 1)
            with torch.no_grad():
                outputs = network(inputs[0])
                # print(f"Outputs shape: {outputs.shape}")
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(outputs.cpu().numpy())
                
        # Calculate R² and RMSE
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        cor, _ = pearsonr(np.array(y_true).reshape(-1), np.array(y_pred).reshape(-1))
        print(f"FOLD {fold},Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}, Correlation: {cor:.4f}, R²: {r2:.4f}, RMSE: {rmse:.4f}")
        
        if r2 > max_r2:
            torch.save(network.state_dict(), f'./Compare/regression_model_fold_{fold}.pt')
            max_r2 = r2
            max_rmse = rmse  
            max_cor = cor 
            print("Max R² increased:", max_r2)

    list_r2.append(max_r2)
    list_rmse.append(max_rmse)
    list_cor.append(max_cor)

mean_r2 = np.mean(list_r2)
mean_rmse = np.mean(list_rmse)
mean_cor = np.mean(list_cor)

print("mean_r2 =", mean_r2)
print("mean_rmse =", mean_rmse)
print("mean_cor =", mean_cor)
print("r2 list:", list_r2)
print("rsme list:", list_rmse)
print("cor list:", list_cor)

print("learning rate: ", lr)
print('Training Finished!')
