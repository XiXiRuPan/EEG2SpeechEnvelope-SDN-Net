import os
import torch
import torch.nn as nn
import pdb
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import json
import scipy
import torch
import torch.nn as nn
import torch

from scipy.stats import spearmanr
class CLUB(nn.Module):  # CLUB: Mutual Information Contrastive Learning Upper Bound
    '''
        This class provides the CLUB estimation to I(X,Y)
        Method:
            forward() :      provides the estimation with input samples  
            loglikeli() :   provides the log-likelihood of the approximation q(Y|X) with input samples
        Arguments:
            x_dim, y_dim :         the dimensions of samples from X, Y respectively
            hidden_size :          the dimension of the hidden layer of the approximation network q(Y|X)
            x_samples, y_samples : samples from X and Y, having shape [sample_size, x_dim/y_dim] 
    '''
    def __init__(self, x_dim, y_dim, hidden_size):
        super(CLUB, self).__init__()
        # p_mu outputs mean of q(Y|X)
        #print("create CLUB with dim {}, {}, hiddensize {}".format(x_dim, y_dim, hidden_size))
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, y_dim))
        # p_logvar outputs log of variance of q(Y|X)
        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, y_dim),
                                       nn.Tanh())

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar
    
    def forward(self, x_samples, y_samples): 
        mu, logvar = self.get_mu_logvar(x_samples)
        
        # log of conditional probability of positive sample pairs
        positive = - (mu - y_samples)**2 /2./logvar.exp()  
        
        prediction_1 = mu.unsqueeze(1)          # shape [nsample,1,dim]
        y_samples_1 = y_samples.unsqueeze(0)    # shape [1,nsample,dim]

        # log of conditional probability of negative sample pairs
        negative = - ((y_samples_1 - prediction_1)**2).mean(dim=1)/2./logvar.exp() 

        return (positive.sum(dim = -1) - negative.sum(dim = -1)).mean()
    def loglikeli(self, x_samples, y_samples): # unnormalized loglikelihood 
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples)**2 /logvar.exp()-logvar).sum(dim=1).mean(dim=0)
    
    def learning_loss(self, x_samples, y_samples):
        return - self.loglikeli(x_samples, y_samples)
# Define the Spearman correlation coefficient loss function
class SPEARMAN(nn.Module):
    def __init__(self):
        super(SPEARMAN, self).__init__()
 
    def forward(self,pred, target):
        pred_ranks = torch.argsort(torch.argsort(pred))
        target_ranks = torch.argsort(torch.argsort(target))
        pred_rank_diff = pred_ranks - target_ranks
        n = pred.shape[0]
        numerator = 6 * torch.sum(pred_rank_diff ** 2)
        denominator = n * (n ** 2 - 1)
        return 1 - (numerator / denominator)

def concordance_correlation_coefficient(predicted, actual):
    # Calculate means
    predicted_mean = torch.mean(predicted)
    actual_mean = torch.mean(actual)

    # Calculate variances
    predicted_var = torch.mean(torch.square(predicted - predicted_mean))
    actual_var = torch.mean(torch.square(actual - actual_mean))

    # Calculate covariances
    covariance = torch.mean((predicted - predicted_mean) * (actual - actual_mean))

    # Calculate Concordance Correlation Coefficient (CCC)
    numerator = 2 * covariance
    denominator = predicted_var + actual_var + torch.square(predicted_mean - actual_mean)
    ccc = numerator / denominator

    return ccc

class CCCLoss(torch.nn.Module):
    def __init__(self):
        super(CCCLoss, self).__init__()

    def forward(self, predicted, actual):
        ccc = concordance_correlation_coefficient(predicted, actual)
        loss = 1 - ccc
        return loss

class PearsonScipy(nn.Module):
   def __init__(self):
       super(PearsonScipy, self).__init__()
   def forward(self,x,y):
       r, p_value = scipy.stats.pearsonr(x, y) 
       loss = 1 - r 
       return loss  #, p_value

class PearsonLoss(nn.Module):
    def __init__(self):
        super(PearsonLoss, self).__init__()

    def forward(self, x, y):
        x = x.view(-1)
        y = y.view(-1)
        vx = x - torch.mean(x)
        vy = y - torch.mean(y)
        pearson_corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
        loss = 1  - pearson_corr
        return loss

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


