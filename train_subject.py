import os
import torch
import torch.nn as nn
import pdb
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR, MultiStepLR, LambdaLR, ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau

import numpy as np
import random
from MLA_Codec_modify import R2AttMLA_Codec
from models.ECAPA_TDNN import ATDNN #ECAPA_TDNN
import json
from dataset.subject_dataset import MyDataset, mycollate, getData

train_data = getData(datapath = "new_workspace//SPEECHEEG/split_data",keyword="train")
val_data = getData(datapath = "new_workspace//SPEECHEEG/split_data",keyword="val")
test_data = getData(datapath = "new_workspace//SPEECHEEG/split_data",keyword="test")

train_dataset = MyDataset(train_data)
val_dataset = MyDataset(val_data)
test_dataset = MyDataset(test_data)
# Create a dataloader
train_dataloader = DataLoader(train_dataset, batch_size=64, collate_fn = mycollate, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)




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
pdb.set_trace()
# create a simple linear model
model = ATDNN(64) #ECAPA_TDNN(64) #R2AttMLA_Codec() #SimpleLinearModel()
model = nn.DataParallel(model.cuda(), device_ids=gpus)
optimizer = torch.optim.Adam(model.parameters(),lr =0.01)
#scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1/(epoch + 1))  
criterion = torch.nn.CrossEntropyLoss()
#criterion = Pearson_loss_cut()

model.train()
num_epochs =100
model_path = os.path.join(results_folder, "model.h5")
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    #maxk =5
    batch_size = target.size(0)
    #batch_size=64
    #
    _, pred = output.topk(maxk, 1, True, True)
    #_,pred::: torch.Size([64, 5]) torch.Size([64, 5])
    #转置操作
    pred = pred.t()
    #pred::: torch.Size([5, 64])
    #torch.Size([5, 64])
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
       # correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        
        #print(correct[:k].view(-1).size())
   
        #correct_k = correct[:k].view(-1).contiguous().float().sum(0, keepdim=True)
        correct_k = correct[:k].reshape(-1).contiguous().float().sum(0, keepdim=True)
        #correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        #correct_k::: tensor([0.], device='cuda:0')
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def train_one_epoch (train_dataloader, state="Train"):
    if state == "Train":
        model.train()
    else: model.eval()
    for eeg, mel, envelop,subject in train_dataloader:
        eeg = eeg.transpose(1,2).cuda(device=torch.device('cuda:0'))
        envelop = envelop.cuda(device=torch.device('cuda:0'))
        mel = mel.cuda(device=torch.device('cuda:0'))
        subject = subject.cuda(device=torch.device('cuda:0'))
        #torch.Size([2, 64, 46148])
        #print("eeg size:",eeg.size())
        
        #print("envelop size",envelop.size())
        
        y_pred = model(eeg)
        
        #print("y_pred size",y_pred.size())
        #print("envelop size",envelop.size())
        loss = criterion(y_pred, subject)
        prec1, prec5 = accuracy(y_pred, subject, topk=(1, 5))
        #print("--------->loss",loss.shape)
        error = loss 
        if state == "Train":
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #scheduler.step()
      
       
    # Print the loss for every 100 epochs
    if (epoch+1) % 2 == 0:
        print(f'{state}: Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Prec1: {prec1.item():.4f}%')
    return loss, error #(error.sum()/error.size()[0]).item()

lowest_error = float("inf")
for epoch in range(num_epochs):
    loss , error = train_one_epoch(train_dataloader)    
    if ((epoch+1) % 2) == 0:
        val_loss , val_error = train_one_epoch(train_dataloader, state="Eval")
        if val_error < lowest_error:
            torch.save(model.state_dict(), "subject_classification_model.pth")
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
        
        print(f'Test: Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Error: {error.item():.4f}')
        
            
            
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
