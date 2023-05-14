import os
import torch
import torch.nn as nn
import pdb
import torch
from torch.utils.data import  DataLoader
from dataset.subject_dataset import getData,MyDataset,mycollate
#from dataset.test_dataset import getData_test,MyDataset_test
from loss.loss import *
import numpy as np
import random
from models import *
import json
import yaml
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

'''Load parameters from config files '''
def load_params(filename):
    f = open(filename, "r")
    conf_str = f.read()
    f.close()
    config = yaml.load(conf_str, Loader=yaml.FullLoader)
    return config
#envelop, mel , eeg dataset preparing 
def prepare_dataset(train_datapath, keyword,batch_size,suffix=".npy"):
    train_data = getData(train_datapath,keyword,suffix)

    train_dataset = MyDataset(train_data,suffix)
    # Create a dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn = mycollate, shuffle=True)
    return train_dataloader

def average_score(dicts):
    inner_key =[]
    for i in range(72): 
        if i > 10:
           inner_key.append("sub-0"+str(i))
        else:
           inner_key.append("sub-00"+str(i))
 
    inner = {}
    out = {}
    for key in dicts.keys():
        if key.split("_")[0] in inner_key:
           if key.split("_")[0]  in inner.keys():
              inner[key].append(dicts[key])
           else: inner[key] = [dicts[key]]
        else:
           if key.split("_")[0]  in out.keys():
              out[key].append(dicts[key])
           else: out[key] = [dicts[key]]
    index = 0
    score = 0
    for key in inner.keys():
        index += 1
        score += sum(inner[key]) / len(inner[key])
    score1 = score / index
    idx = 0
    sc = 0
    for key in out.keys():
        idx += 1    
        sc += sum(out[key]) / len(out[key])
    score2 = sc / idx
    result = score1 *2 / 3 + score2 /3
    return result
# Provide the path of the dataset
# which is split already to train, val, test
#data_folder = os.path.join(config["dataset_folder"], config["split_folder"])
def initial_models(params):
    

    # Create a directory to store (intermediate) results
    results_folder = os.path.join(params['result_folder'])
    os.makedirs(results_folder, exist_ok=True)
    gpus =list(range(params['num_gpus']))
    # create a deep neural network model
    modeli_name = params['model_name']
    if model_name == "SimpleLinearModel":
       model = SimpleLinearModel() 
    elif model_name == "R2AttMLA_Codec":
       model = R2AttMLA_Codec()
    elif model_name == "SATDNN":
       model = SATDNN(64)
    elif model_name == 'MLA_Codec':
       model = MLA_Codec()
    elif model_name == 'R2MLA_Codec':
       model = R2MLA_Codec()
    elif model_name == 'AttMLA_Codec':
       model = AttMLA_Codec()
    elif model_name == 'NestedUNet':
       model = NestedUNet()
    elif model_name == 'ConformerAT':
       model = ConformerAT()
    elif model_name == 'ConformerTDNN':
       model = ConformerTDNN(64)
    elif model_name == 'ConformerModel':
       model = ConformerModel(64)
    model = nn.DataParallel(model.cuda(), device_ids=gpus)
    optimizer = torch.optim.Adam(model.parameters(),lr=params['init_lr'])
    #criterion = PearsonScipy()
    #metric = Pearson_loss_cut()
    #criterion = Pearson_loss_cut()
    #criterion = torch.nn.MSELoss()
    criterion = torch.nn.CrossEntropyLoss()
    #criterion = PearsonLoss()
    #criterion = SPEARMAN()#CCCLoss() 
    metric = PearsonLoss()
    return model, optimizer, criterion,metric
'''
    model.train()
    
       model = ConformerTDNN()num_epochs =100
    model_path = os.path.join(results_folder, "model.h5")
'''

def adjust_learning_rate(params,optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = params['init_lr'] * (0.1 ** (epoch // params['update_step']))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr




def train_model(params,train_dataloader,test_dataloader):
    #
    model, optimizer, criterion,metric = initial_models(params)

    accuracy_high = 0.0
    num_epochs = params['num_epochs']
    for epoch in range(num_epochs):
        model.train()
        index = 0
        adjust_learning_rate(params,optimizer, epoch)
        #for eeg, mel, envelop in train_dataloader:
        for eeg, envelop, subject in train_dataloader:
            lens = len(train_dataloader)
            index += 1
            #eeg = eeg.cuda(device=torch.device('cuda:0'))
            eeg = eeg.transpose(1,2).cuda(device=torch.device('cuda:0'))
            envelop = envelop.cuda(device=torch.device('cuda:0'))
            subject = subject.cuda(device=torch.device('cuda:0'))
            '''
            # Normalize data
            data_mean = np.expand_dims(np.mean(sub_data, axis=1), axis=1)
            data_std = np.expand_dims(np.std(sub_data, axis=1), axis=1)
            sub_data = (sub_data - data_mean) / data_std
            '''
            y_pred = model(eeg)
            pdb.set_trace()
            loss = criterion(y_pred, subject)
            prec1, prec5 = accuracy(y_pred, subject, topk=(1, 5))
            '''
            y_pred = model(eeg)
            y_pred = y_pred.transpose(1,2)
            loss = criterion(envelop.float(), y_pred.float())
            
            #loss = (loss_cc + peloss)/2
            error = -(metric(envelop, y_pred)) + 1
            '''
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f'Train: Epoch [{epoch+1}/{num_epochs}], Batch [{index} /{lens}], Loss: {loss.item():.4f}, Prec1: {prec1.item():.4f}')
            if epoch % 10 == 0:
               torch.save(model.state_dict(), params['result_folder']+"/middle_"+str(epoch)+".pth")

        model.eval()
        dicts={}
        index = 0
        if (epoch +1) % 50 ==0:     
            with torch.no_grad():
                for eeg, envelop,subject in test_dataloader:
                    index += 1
                    lens = len(test_dataloader)
                    '''
                    eeg_mean = torch.mean(eeg,dim=1,keepdim=True)
                    eeg_var = torch.var(eeg, dim =1,keepdim=True)
                    eeg = (eeg - eeg_mean) / eeg_var
                    '''
                    eeg = eeg.transpose(1,2).cuda(device=torch.device('cuda:0'))
                    #eeg = eeg.cuda(device=torch.device('cuda:0'))
                    envelop = envelop.cuda(device=torch.device('cuda:0'))
                    subject = subject.cuda(device=torch.device('cuda:0'))
                    y_pred = model(eeg)
                    loss = criterion(y_pred, subject)
                    prec1, prec5 = accuracy(y_pred, subject, topk=(1, 5))

                    print(f'Validate: Epoch [{epoch+1}/{num_epochs}], Batch [{index} /{lens}], Loss: {loss.item():.4f}, Prec1: {prec1.item():.4f}')
                    if prec1 > accuracy_high:
                       torch.save(model.state_dict(), params['result_folder']+"/best_validate_best_"+str(epoch)+".pth")
                       accuracy_high = prec1
                #results = average_score(dicts)
                #print(f'Validate: Epoch [{epoch+1}/{num_epochs}], Results: {results}')


if __name__ =='__main__':
   import time 
   print (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) )   
   config_file = 'params/subject_config.yaml' 
   params = load_params(config_file) 
   model_name = params['model_name']
   train_dataset = prepare_dataset(params['train_datapath'],params['train_keyword'],params['batch_size'])
   test_dataset = prepare_dataset(params['test_datapath'],params['test_keyword'],params['batch_size'],suffix=".json")
   #test_dataset= MyDataset_test(test_data)

   #test_dataloader = DataLoader(test_dataset, batch_size=64) # collate_fn = mycollate, shuffle=True)
   train_model(params, train_dataset,test_dataset)
   print (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) )   
