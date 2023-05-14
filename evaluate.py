import os
import torch
import torch.nn as nn
import pdb
import torch
from torch.utils.data import  DataLoader

from dataset.evaluation_dataset import getData,MyDataset
from loss.loss import *
import numpy as np
import random
from models import *
import json
import yaml
import pdb
'''Load parameters from config files '''
def load_params(filename):
    f = open(filename, "r")
    conf_str = f.read()
    f.close()
    config = yaml.load(conf_str, Loader=yaml.FullLoader)
    return config

#envelop, mel , eeg dataset preparing 
def prepare_dataset(test_datapath, keyword,batch_size):
    test_data = getData(test_datapath,keyword)

    test_dataset = MyDataset(test_data)
    # Create a dataloader
    train_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    return train_dataloader


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
       model = nn.DataParallel(model.cuda(), device_ids=gpus)
       model.load_state_dict(torch.load(params['best_model']))
 
    elif model_name == "R2AttMLA_Codec":
       model = R2AttMLA_Codec()
       model = nn.DataParallel(model.cuda(), device_ids=gpus)
       model.load_state_dict(torch.load(params['best_model']))
    elif model_name == "ATDNN":
       #pdb.set_trace()
       model = ATDNN(64)
       model = nn.DataParallel(model.cuda(), device_ids=gpus)
       model.load_state_dict(torch.load(params['best_model']))

    model = nn.DataParallel(model.cuda(), device_ids=gpus)

    # criterion = Pearson_loss_cut()
    criterion = PearsonLoss()
    return model,  criterion
'''
    model.train()
    num_epochs =100
    model_path = os.path.join(results_folder, "model.h5")
'''

def evaluate_model (test_dataloader, model, criterion, state="Test"):
    model.eval()
    dicts = {}
    inner_subject = {}
    out_subject = {}
    inners = []
    for i in range(1,72):
        if i < 10: 
           inners.append("sub-00"+str(i))
        else:
           inners.append("sub-0"+str(i))
    for eeg,  envelop,subject,theme in test_dataloader:
        sub_dict = {}
        eeg = eeg.transpose(1,2).cuda(device=torch.device('cuda:0'))
        envelop = envelop.cuda(device=torch.device('cuda:0'))
        #pdb.set_trace() 
        y_pred = model(eeg)
        y_pred = y_pred.transpose(1,2)
        error = -criterion(envelop, y_pred) #+ 1

        if subject[0].split("_")[0] in inners:
           if subject[0].split("_")[0] in inner_subject.keys():
              inner_subject[subject[0].split("_")[0]].append( error.item())
           else: inner_subject[subject[0].split("_")[0]] = [error.item()]
        else: 
           if subject[0].split("_")[0] in out_subject.keys():
              out_subject[subject[0].split("_")[0]].append( error.item())
           else: out_subject[subject[0].split("_")[0]] = [error.item()]
   
           
        y_pred = y_pred.squeeze().tolist()

        sub_dict[subject] = [y_pred, theme]
        name = subject[0].split("_")[0]
        if name in dicts.keys():
           dicts[name].append(sub_dict)
        else: dicts[name] = [sub_dict]
        
        print(f'{state}:{subject} {error.item():.4f}')
    inner_score  = 0
    index = 0
    for key in inner_subject.keys():
        index = index + 1
        scores = inner_subject[key]
        score = sum(scores) / len(scores)
        inner_score +=  score
    print("index--->:",index)
    inner_score = inner_score / index
    out_score = 0
    index1=0
    for key in out_subject.keys():
        index1 += 1
        scores = out_subject[key]
        score = sum(scores) / len(scores)
        out_score +=  score
    print("index1--->:",index1)
    out_score = out_score / 14
    #pdb.set_trace() 
    return inner_score * 2 / 3 + out_score * 1 / 3, dicts




if __name__ =='__main__':
   config_file = 'params/test.yaml' 
   params = load_params(config_file) 
   model_name = params['model_name']
   test_dataset = prepare_dataset(params['test_eeg'],params['test_label'],params['batch_size'])
   model,  criterion = initial_models(params)
   r1, dicts =evaluate_model(test_dataset,model,criterion)
   decode_path=params['results_decoding']

   os.makedirs(decode_path, exist_ok=True)
   print("score:::",r1)
   for key in dicts:
       each_subject = {}
       save_path =decode_path+"/"
       os.makedirs(save_path, exist_ok=True)
       for item in dicts[key]:
           for kk in item.keys():
               #pdb.set_trace()
               each_subject[kk[0]] = [item[kk][0],item[kk][1][0]]
            
       with open(save_path+"/"+key+".json","w") as fp: 
            json.dump(each_subject, fp)
