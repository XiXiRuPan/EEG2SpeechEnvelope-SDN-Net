#test 
import os
import torch
import torch.nn as nn
import pdb
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from umodel_modify import R2AttMLA_Codec
import json
from SimpleLinearModel import SimpleLinearModel

from loss.loss import Pearson_loss_cut

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
    min_lens = 64

    
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


gpus =[0,1]
#model = SimpleLinearModel()
test_data = getData(datapath = "/mnt/worknfs/xxx/new_workspace//SPEECHEEG/split_data",keyword="test")

test_dataset = MyDataset(test_data)
# Create a dataloader
test_dataloader = DataLoader(test_dataset, batch_size=2, collate_fn = mycollate, shuffle=True)
model = R2AttMLA_Codec()

model = nn.DataParallel(model.cuda(), device_ids=gpus)
model.load_state_dict(torch.load('results/vlaai/lowest_error_model.pth'))
model.eval()
criterion =Pearson_loss_cut() 
metric = Pearson_loss_cut() 

def evaluate_epoch (test_dataloader ):
    model.eval()
    evaluation = {}
    index = 0
    for eeg, mel, envelop in test_dataloader:
        eeg = eeg.transpose(1,2).cuda(device=torch.device('cuda:0'))
        mel = mel.cuda(device=torch.device('cuda:0'))
        envelop = envelop.cuda(device=torch.device('cuda:0'))
        index = index + 1
        #torch.Size([2, 64, 46148])
        #print("eeg size:",eeg.size())
        
        #print("envelop size",envelop.size())
        
        y_pred = model(eeg)
        y_pred = y_pred.transpose(1,2)
        #print("y_pred size",y_pred.size())
     
        loss = criterion(envelop,y_pred)
        
        error = - metric(envelop, y_pred)
        loss = loss.squeeze()
        error = error.squeeze()
        #print("loss.shape",loss.shape)
        #print( loss.squeeze().shape)
        #print(error.squeeze().shape)
   
       
        
        losses = []
        errors = []
        for idx in range(len(loss)):
            
            losses.append(loss[idx].item())
            errors.append(error[idx].item())
        
        evaluation[str(index)] = dict(zip(losses, errors))   
 
        
        print(f'Test: batch [{index}/{len(test_dataloader)}], Loss: {(loss.sum()/ loss.size()[0]).item():.4f}, Error: {(error.sum()/error.size()[0]).item():.4f}')
        
            
    print("---->loss.average:",sum(losses)/len(losses))        
    return evaluation

pdb.set_trace()
# Evaluate the model
evaluation =  evaluate_epoch(test_dataloader)
 
results_path = os.path.join("", "results.json")
print(type(evaluation))
print(evaluation.keys())
print(evaluation['1'])
with open(results_path, "w") as fp:
    json.dump(evaluation, fp)

