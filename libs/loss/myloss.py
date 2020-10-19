import torch
import torch.nn as nn
import torch.nn.functional as F

loss_l2 = nn.MSELoss()

class myLossD(nn.Module):
    def __init__(self, ignore_index=255):
        super(myLossD, self).__init__()
        self.ignore_index = ignore_index
        self.gamma = 2
    
    def forward(self, input, target):
        """
        Arguments:
            input: (B, C, H, W)

        Return:
            loss
        """
        # (B, _, H, W) = input.shape
        # import pdb; pdb.set_trace()
        loss = 0
        for pred in input:
            labels = torch.ones(pred.shape).cuda()*target
            loss += F.binary_cross_entropy_with_logits(pred, labels)
        return loss

class myLoss(nn.Module):
    def __init__(self, ignore_index=255):
        super(myLoss, self).__init__()
        self.ignore_index = ignore_index
        self.gamma = 2
    
    def forward(self, input, target):
        """
        Arguments:
            input: (B, C, H, W)

        Return:
            loss
        """
        # (B, _, H, W) = input.shape
        preds = input
        labels = target
        preds_softmax = F.softmax(preds, dim=1)
        preds_logsoft = torch.log(preds_softmax)
        effieicent = torch.pow((1-preds_softmax),self.gamma)
        # loss = F.nll_loss(20*effieicent*preds_logsoft, target.long(), ignore_index=self.ignore_index, reduction='none')
        loss = F.nll_loss(preds_logsoft, target.long(), ignore_index=self.ignore_index, reduction='none')
        loss = torch.mean(loss)
        return loss

class myLoss2(nn.Module):
    def __init__(self, ignore_index=255):
        super(myLoss2, self).__init__()
        self.ignore_index = ignore_index
        self.gamma = 2
    
    def forward(self, input, target):
        """
        Arguments:
            input: (B, C, H, W)

        Return:
            loss
        """
        # (B, _, H, W) = input.shape
        preds = input
        labels = target
        eff1 = torch.pow((2-preds),self.gamma)
        eff2 = torch.pow(1+preds,self.gamma)
        # loss = -(target*torch.log(preds+1e-9)*eff1+(1-target)*torch.log(1-preds+1e-9)*eff2).mean()
        loss = -((target==1).float()*torch.log(preds+1e-9)*eff1+(target==0).float()*torch.log(1-preds+1e-9)*eff2).sum()/((target==1).sum()+(target==0).sum())
        return loss
class myLoss3(nn.Module):
    def __init__(self, ignore_index=255):
        super(myLoss3, self).__init__()
        self.ignore_index = ignore_index
        self.gamma = 2
    
    def forward(self, input, target):
        """
        Arguments:
            input: (B, C, H, W)

        Return:
            loss
        """
        # (B, _, H, W) = input.shape
        preds = input
        labels = target
        # import pdb; pdb.set_trace()
        zero_sum = (labels==0).sum()
        one_sum = (labels==1).sum()+1
        # w_zero = one_sum/zero_sum
        w_one = zero_sum/one_sum
        eff1 = torch.pow((1-preds),self.gamma)
        eff2 = torch.pow(preds,self.gamma)
        loss = -((target==1).float()*torch.log(preds+1e-9)*eff1*w_one+(target==0).float()*torch.log(1-preds+1e-9)*eff2).mean()
        return loss*2