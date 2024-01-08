import torch
from torch.nn import functional as F
import torchattacks 

from model.modelBuilder import GradCamMod
from utils import get_img_norm_params

class Loss(torch.nn.Module):
    def __init__(self,args,reduction="mean",net=None):
        super(Loss, self).__init__()
        self.reduction = reduction
        self.args= args

        assert (args.nll_weight > 0) or (args.focal_weight > 0),"Choose one loss function"
        assert not (args.nll_weight > 0 and args.focal_weight > 0),"Cannot use the two loss functions"

        if args.adv_ce_weight > 0:
            net_gradcammod = GradCamMod(net)
            self.atk = torchattacks.PGD(net_gradcammod, eps=args.maximum_pert/255, alpha=2/255, steps=4)
            mean_std_dict = get_img_norm_params(args.dataset_train)
            self.atk.set_normalization_used(mean=mean_std_dict["mean"], std=mean_std_dict["std"])

    def forward(self,output,target,resDict):
        return computeLoss(self.args,output, target, resDict,reduction=self.reduction)

def computeLoss(args, output, target, resDict,reduction="mean"):
    loss_dic = {}

    loss = 0

    if args.focal_weight > 0:
        loss_func = adaptive_focal_loss
        weight = args.focal_weight if args.adv_ce_weight <= 0 else 0
        key = "focal_loss"
    else:
        loss_func = F.cross_entropy
        weight = args.nll_weight if args.adv_ce_weight <= 0 else 0
        key = "loss_ce"
          
    loss,loss_dic = add_loss_term(loss,key,loss_func,output,target,reduction,loss_dic,weight)

    if args.adv_ce_weight > 0 and "output_adv" in resDict:
        loss,loss_dic = add_loss_term(loss,"loss_adv",loss_func,resDict["output_adv"],target,reduction,loss_dic,args.adv_ce_weight)

    if args.loss_on_masked:
        if args.adv_ce_weight > 0:
            weight_masked = 0
        elif args.focal_weight > 0:
            weight_masked = args.focal_weight
        else:
            weight_masked = args.nll_weight

        loss,loss_dic = add_loss_term(loss,"loss_masked",loss_func,resDict["output_masked"],target,reduction,loss_dic,weight_masked)

        if args.adv_ce_weight > 0 and "output_masked_adv" in resDict:
            loss,loss_dic = add_loss_term(loss,"loss_masked_adv",loss_func,resDict["output_masked_adv"],target,reduction,loss_dic,args.adv_ce_weight)
            
    loss_dic["loss"] = loss.unsqueeze(0)

    return loss_dic

def add_loss_term(loss,key,loss_func,output,target,reduction,loss_dic,weight):
    loss_term = loss_func(output, target,reduction=reduction)
    loss_dic[key] = loss_term.data.unsqueeze(0)
    loss += weight*loss_term
    return loss,loss_dic

'''
Implementation of Focal Loss with adaptive gamma.
Reference:
[1]  T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollar, Focal loss for dense object detection.
     arXiv preprint arXiv:1708.02002, 2017.
'''
def get_gamma_dic():
    return {0.2:5.0,0.5:3.0,1:1}

def get_gamma_list(pt):
    gamma_dict = get_gamma_dic()
    gamma_list = []
    batch_size = pt.shape[0]
    for i in range(batch_size):
        pt_sample = pt[i].item()

        j = 0
        gamma_found =False
        thres_list = list(sorted(gamma_dict.keys()))
        while (not gamma_found) and (j < len(thres_list)):

            gamma = gamma_dict[thres_list[j]]
            gamma_found = pt_sample < thres_list[j]

            if gamma_found:
                gamma_list.append(gamma)
            else:
                j += 1
        
        if not gamma_found:
            gamma_list.append(gamma_dict[thres_list[-1]])

    return torch.tensor(gamma_list).to(pt.device)

def adaptive_focal_loss(logits, target,reduction):

    target = target.view(-1,1)
    logpt = F.log_softmax(logits, dim=1).gather(1,target).view(-1)
    pt = F.softmax(logits, dim=1).gather(1,target).view(-1)

    gamma = get_gamma_list(pt)
    loss = -1 * (1-pt)**gamma * logpt

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        raise ValueError("Unkown reduction method",reduction)

def agregate_losses(loss_dic):
    for loss_name in loss_dic:
        loss_dic[loss_name] = loss_dic[loss_name].sum()
    return loss_dic