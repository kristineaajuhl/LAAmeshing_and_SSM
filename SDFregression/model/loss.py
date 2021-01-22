import torch.nn.functional as F
import torch
from scipy import ndimage
import numpy as np



# Negative log-likelyhood
def nll_loss(output, target):
    return F.nll_loss(output, target)

# Mean-squared-error
def mse_loss(output, target):
    return F.mse_loss(output, target)

# Binary cross entropy
def binary_cross_entropy(output, target):
    #weight = target+1
    return F.binary_cross_entropy(output, target)

def binary_cross_entropy_with_logits(output,target):
    return F.binary_cross_entropy_with_logits(output,target)

# Cross entropy 
def cross_entropy(output,target):
    return F.cross_entropy(output,target)

def dice_loss(output,target):
    smooth = 1
    output = F.sigmoid(output)       
    
    #flatten label and prediction tensors
    output = output.contiguous().view(-1)
    target = target.contiguous().view(-1)
    
    intersection = (output * target).sum()                            
    dice = (2.*intersection + smooth)/(output.sum() + target.sum() + smooth)  
    
    return 1 - dice

def sdf_regularized_BCE(output,target,weight):
    return F.binary_cross_entropy_with_logits(output,target,1+weight)

def weighted_mse_loss(output,target):
    ret = 1 / (abs(target) + 0.001) * (output - target) ** 2
    ret = torch.mean(ret)

    return ret