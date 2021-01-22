import torch

# DICE for binary segmentation only
def dice(output, target):
    with torch.no_grad():
        pred = torch.argmax(torch.nn.functional.softmax(output,dim=1),dim=1)
        #pred = torch.argmax(output, dim=1)
        #pred = output
        lab = torch.argmax(target, dim=1)
        #lab = target
        assert pred.shape == lab.shape
        
        X = 2*lab-pred
        
        TP = torch.sum(X == 1).item()
        FP = torch.sum(X == 2).item()
        FN = torch.sum(X == -1).item()
        
        dice = 2*TP/(2*TP+FP+FN)
    return dice

#whole heart dice (multiclass)
def whDice(output,target):
    with torch.no_grad():
        pred = torch.argmax(torch.nn.functional.softmax(output,dim=1),dim=1)>0
        lab = torch.argmax(target, dim=1)>0
        
        assert pred.shape == lab.shape
        
        X = 2*lab-pred
        
        TP = torch.sum(X == 1).item()
        FP = torch.sum(X == 2).item()
        FN = torch.sum(X == -1).item()
        
        dice = 2*TP/(2*TP+FP+FN)
    return dice


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(torch.nn.functional.softmax(output,dim=1),dim=1)
        #pred = torch.argmax(output, dim=1)
        #pred = output
        lab = torch.argmax(target, dim=1)
        #lab = target
        assert pred.shape == lab.shape
        correct = torch.sum(pred == lab).item()
    return correct / torch.numel(lab)