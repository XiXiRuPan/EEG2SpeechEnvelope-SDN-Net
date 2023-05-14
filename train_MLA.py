import os
import torch
import torch.nn as nn
import pdb
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR, MultiStepLR, LambdaLR, ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau

import numpy as np
import random
from umodel_modify import R2AttMLA_Codec
import json

def pearson_tf(y_true, y_pred, axis=1):
    # Compute the mean of the true and predicted values
    y_true_mean = torch.mean(y_true, axis, keepdim=True)
    y_pred_mean = torch.mean(y_pred, axis, keepdim=True)

    # Compute the numerator and denominator of the pearson correlation
    numerator = torch.sum(
        (y_true - y_true_mean) * (y_pred - y_pred_mean),
        axis,
        keepdim=True,
    )
    std_true = torch.sum(torch.square(y_true - y_true_mean), axis, keepdim=True)
    std_pred = torch.sum(torch.square(y_pred - y_pred_mean), axis, keepdim=True)
    denominator = torch.sqrt(std_true * std_pred)

    # Compute the pearson correlation
    return torch.div(numerator, denominator)

def correlation(x, y, eps=1e-8):
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)

    corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + eps)
    return corr
 

import torch

def pearson_correlation(y_true, y_pred, axis=1):
    """
    Computes the Pearson correlation coefficient between two tensors.

    Parameters
    ----------
    y_true: torch.Tensor
        Ground truth labels. Shape is (batch_size, time_steps, n_features)
    y_pred: torch.Tensor
        Predicted labels. Shape is (batch_size, time_steps, n_features)
    axis: int
        Axis along which to compute the pearson correlation. Default is 1.

    Returns
    -------
    torch.Tensor
        Pearson correlation coefficient.
        Shape is (batch_size, 1, n_features) if axis is 1.
        x.size()--> torch.Size([128, 6194, 1])
    y_pred.size()--> torch.Size([128, 6194, 1])
    pearson_corr.size()--> torch.Size([128, 1, 1])
    """
    #print("x.size()-->",y_true.size())
    #print("y_pred.size()-->",y_pred.size())
    # Compute the mean of the true and predicted values
    y_true_mean = torch.mean(y_true, dim=axis, keepdim=True)
    y_pred_mean = torch.mean(y_pred, dim=axis, keepdim=True)

    # Compute the numerator and denominator of the pearson correlation
    numerator = torch.sum((y_true - y_true_mean) * (y_pred - y_pred_mean), dim=axis, keepdim=True)
    std_true = torch.sqrt(torch.sum(torch.square(y_true - y_true_mean), dim=axis, keepdim=True))
    std_pred = torch.sqrt(torch.sum(torch.square(y_pred - y_pred_mean), dim=axis, keepdim=True))
    denominator = std_true * std_pred
    
    # Compute the pearson correlation
    pearson_corr = torch.div(numerator, denominator)
    #print("pearson_corr.size()-->",pearson_corr.size())
    return pearson_corr

def pearson_loss(y_true, y_pred, axis=1):
    # Compute the negative pearson correlation
    return -pearson_tf(y_true, y_pred, axis)

def pearson_metric(y_true, y_pred, axis=1):
    """Pearson metric function.

    Parameters
    ----------
    y_true: torch.Tensor
        True values. Shape is (batch_size, time_steps, n_features)
    y_pred: torch.Tensor
        Predicted values. Shape is (batch_size, time_steps, n_features)

    Returns
    -------
    torch.Tensor
        Pearson metric.
        Shape is (batch_size, 1, n_features)
    """
    # Compute the mean of the true and predicted values
    y_true_mean = torch.mean(y_true, dim=axis, keepdim=True)
    y_pred_mean = torch.mean(y_pred, dim=axis, keepdim=True)

    # Compute the numerator and denominator of the pearson correlation
    numerator = torch.sum(
        (y_true - y_true_mean) * (y_pred - y_pred_mean),
        dim=axis,
        keepdim=True,
    )
    std_true = torch.sum(torch.square(y_true - y_true_mean), dim=axis, keepdim=True)
    std_pred = torch.sum(torch.square(y_pred - y_pred_mean), dim=axis, keepdim=True)
    denominator = torch.sqrt(std_true * std_pred)

    # Compute the pearson correlation
    return torch.div(numerator, denominator)

class Pearson_loss_cut(nn.Module):
    def forward(self,y_true, y_pred, axis=1):
        """Pearson loss function.

        Parameters
        ----------
        y_true: torch.Tensor
            True values. Shape is (batch_size, time_steps, n_features)
        y_pred: torch.Tensor
            Predicted values. Shape is (batch_size, time_steps, n_features)

        Returns
        -------
        torch.Tensor
            Pearson loss.
            Shape is (batch_size, 1, n_features)
        """
        # y_true t_pred

        return -pearson_tf(y_true[:, : y_pred.shape[1], :], y_pred, axis=axis)
        #return -pearson_correlation(y_true[:, : y_pred.shape[1], :], y_pred)


class  Pearson_metric_cut(nn.Module):
    
    def forward(self, y_true, y_pred, axis=1):
        """Pearson metric function.

        Parameters
        ----------
        y_true: torch.Tensor
            True values. Shape is (batch_size, time_steps, n_features)
        y_pred: torch.Tensor
            Predicted values. Shape is (batch_size, time_steps, n_features)

        Returns
        -------
        torch.Tensor
            Pearson metric.
            Shape is (batch_size, 1, n_features)
        """
    
        return pearson_tf(y_true[:, : y_pred.shape[1], :], y_pred, axis=axis)
        #return pearson_correlation(y_true[:, : y_pred.shape[1], :], y_pred)


print("finish running")

# dataset and dataloader construct [train, vak, test]
def getData(datapath = "/kaggle/input/eeg-speech-splitdata/split_data",keyword="test"):
    train_eeg = []
    train_mel = []
    train_envelope = []
    for root, dirs, files in os.walk(datapath):
        for file in files:
            if ".npy" in file and keyword in file:
                if "eeg" in file:
                    eeg_file = root+"/"+file
                    mel_file = eeg_file.replace("eeg.npy","mel.npy")
                    envelope_file = eeg_file.replace("eeg.npy","envelope.npy")
                    #print(eeg_file)
                    #print(mel_file)
                    #print(envelope_file)
                    if os.path.exists(eeg_file) and  os.path.exists(mel_file) and os.path.exists(envelope_file):
                        
                        train_eeg.append(eeg_file)
                        train_mel.append(mel_file)
                        train_envelope.append(envelope_file)
                        #train_envelope.append(root+"/"+file)
       
    return (train_eeg, train_mel, train_envelope)
               
# Define a custom dataset
class MyDataset(Dataset):
    def __init__(self,train_data ):
        train_eeg, train_mel, train_envelop = train_data
        self.eeg = train_eeg
        self.mel = train_mel
        self.envelop = train_envelop
        
    def __len__(self):
        return len(self.eeg)
    
    def __getitem__(self, index):
  
        eeg  = self.eeg[index]
        mel =  self.mel[index]
        envelop =self.envelop[index]
        eeg_tensor = torch.tensor(np.load(eeg))
        mel_tensor = torch.tensor(np.load(mel), requires_grad=True)
        envelop = torch.tensor(np.load(envelop),requires_grad=True)
     
        return eeg_tensor, mel_tensor, envelop
    
def mycollate(batch):
    
    lens = []
    max_lens = 0
    for item in batch:
        lens.append(item[0].shape[0])
  
    min_lens = min(lens)
    min_lens = min(min_lens,2560)
 
    #min_lens = 641
    eeg_new = []
    mel_new = []
    envelop_new = []
    
    for item in batch:
        random_start = random.randint(0,item[0].shape[0]-min_lens)
        random_end = random_start + min_lens 
        eeg_new.append(item[0][random_start:random_end])
     
        mel_new.append(item[1][random_start:random_end])
        envelop_new.append(item[2][random_start:random_end])
    eeg = torch.stack(eeg_new)
    mel = torch.stack(mel_new)
    envelop = torch.stack(envelop_new)
      
    return  eeg, mel, envelop

train_data = getData(datapath = "/mnt/worknfs/xxx/new_workspace//SPEECHEEG/split_data",keyword="train")
val_data = getData(datapath = "/mnt/worknfs/xxx/new_workspace//SPEECHEEG/split_data",keyword="val")
test_data = getData(datapath = "/mnt/worknfs/xxx/new_workspace//SPEECHEEG/split_data",keyword="test")

train_dataset = MyDataset(train_data)
val_dataset = MyDataset(val_data)
test_dataset = MyDataset(test_data)
# Create a dataloader
train_dataloader = DataLoader(train_dataset, batch_size=128, collate_fn = mycollate, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=True)




# Parameters
# Length of the decision window
window_length = 10 * 64  # 10 seconds
# Hop length between two consecutive decision windows
hop_length = 64
epochs = 100
patience = 5
batch_size = 64
only_evaluate = False
training_log_filename = "training_log.csv"
results_filename = 'eval.json'
'''

# Get the path to the config gile
experiments_folder = os.path.dirname(__file__)
task_folder = os.path.dirname(experiments_folder)
config_path = os.path.join(task_folder, 'util', 'config.json')


# Load the config
with open(config_path) as fp:
    config = json.load(fp)
''' 
# Provide the path of the dataset
# which is split already to train, val, test
#data_folder = os.path.join(config["dataset_folder"], config["split_folder"])
experiments_folder="/kaggle/working/"
stimulus_features = ["envelope"]

features = ["eeg"] + stimulus_features

# Create a directory to store (intermediate) results
results_folder = os.path.join(experiments_folder, "results_linear_baseline")
os.makedirs(results_folder, exist_ok=True)
gpus =[0,1]
# create a simple linear model
model = R2AttMLA_Codec() #SimpleLinearModel()
model = nn.DataParallel(model.cuda(), device_ids=gpus)
optimizer = torch.optim.Adam(model.parameters(),lr =0.01)
#scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1/(epoch + 1))  

criterion = Pearson_loss_cut()

metric = Pearson_metric_cut()
model.train()
num_epochs =100
model_path = os.path.join(results_folder, "model.h5")

def train_one_epoch (train_dataloader, state="Train"):
    if state == "Train":
        model.train()
    else: model.eval()
   
    for eeg, mel, envelop in train_dataloader:
        eeg = eeg.transpose(1,2).cuda(device=torch.device('cuda:0'))
        envelop = envelop.cuda(device=torch.device('cuda:0'))
        mel = mel.cuda(device=torch.device('cuda:0'))
        #torch.Size([2, 64, 46148])
        #print("eeg size:",eeg.size())
        
        #print("envelop size",envelop.size())
        
        y_pred = model(eeg)
        y_pred = y_pred.transpose(1,2)
        #print("y_pred size",y_pred.size())
        #print("envelop size",envelop.size())
        loss = criterion(envelop, y_pred)
        #print("--------->loss",loss.shape)
        error = metric(envelop, y_pred)
        loss = loss.sum() / loss.size()[0]
      
        if state == "Train":
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #print("第%d个epoch的学习率：%f" % (epoch, optimizer.param_groups[0]['lr']))
            #scheduler.step()
      
       
    # Print the loss for every 100 epochs
    if (epoch+1) % 2 == 0:
        print(f'{state}: Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Error: {(error.sum()/error.size()[0]).item():.4f}')
    return loss, (error.sum()/error.size()[0]).item()

lowest_error = float("inf")
for epoch in range(num_epochs):
    loss , error = train_one_epoch(train_dataloader)    
    if ((epoch+1) % 2) == 0:
        val_loss , val_error = train_one_epoch(train_dataloader, state="Eval")
        if val_error < lowest_error:
            torch.save(model.state_dict(), "umodel_lowest_error_model.pth")
            lowest_error = val_error
    
#test 
model = SimpleLinearModel()
model.load_state_dict(torch.load('/lowest_error_model.pth'))
model.eval()
 
def evaluate_epoch (test_dataloader ):
    model.eval()
    evaluation = {}
    index = 0
    for eeg, mel, envelop in test_dataloader:
        eeg = eeg.transpose(1,2)
        index = index + 1
        #torch.Size([2, 64, 46148])
        #print("eeg size:",eeg.size())
        
        print("envelop size",envelop.size())
        
        y_pred = model(eeg)
        y_pred = y_pred.transpose(1,2)
        print("y_pred size",y_pred.size())
     
        loss = criterion(envelop,y_pred)
        
        error = metric(envelop, y_pred)
        loss = loss.squeeze()
        error = error.squeeze()
        print("loss.shape",loss.shape)
        print( loss.squeeze().shape)
        print(error.squeeze().shape)
        for l in loss:
            print(l)
        evaluation[str(index)] = dict(zip(loss, error))
        loss = loss.sum()
        error = error.sum()
        losses = []
        errors = []
        for idx in range(len(loss)):
            losses.append(loss[idx].item())
            errors.append(error[idx].item())
            
        evaluation[str(index)] = dict(zip( losses, errors))
        
        print(f'Test: Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Error: {error.sum().item():.4f}')
        
            
            
    return evaluation


# Evaluate the model
evaluation =  evaluate_epoch(train_dataloader)
 
results_path = os.path.join("", "results.json")
print(type(evaluation))
print(evaluation.keys())
print(evaluation['1'])
with open(results_path, "w") as fp:
    json.dump(evaluation, fp)
logging.info(f"Results saved at {results_path}")
