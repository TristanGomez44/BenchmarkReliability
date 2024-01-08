import numpy as np
import torch 
from math import ceil

def acc_and_ece_selection(path,topk_perc=0.1):
    print("Path",path)
    arr = np.genfromtxt(path,delimiter=",",dtype=str)

    header = arr[0]

    acc_col = np.where(header=="accuracy")[0][0]
    ece_col = np.where(header=="ece")[0][0]
    epoch_col = np.where(header=="epoch")[0][0]
    
    arr = arr[1:].astype("float")
    arr = torch.from_numpy(arr)

    acc = arr[:,acc_col]
    ece = arr[:,ece_col]
    epochs = arr[:,epoch_col]

    return _acc_and_ece_selection(acc,ece,epochs,topk_perc)

def _acc_and_ece_selection(acc,ece,epochs,topk_perc=0.1,minimum_top_x_nb=5):

    top_x_nb = max(ceil(topk_perc * len(acc)),minimum_top_x_nb)
    top_x_nb = min(top_x_nb,len(acc))
    _,best_acc_indices = torch.topk(acc,k=top_x_nb)
    selected_ece_values = ece[best_acc_indices]
    best_epoch_inds = torch.where(ece == selected_ece_values.min())[0]
    if len(best_epoch_inds)>1:
        print("Warning: several epochs yield the best performance. Choosing the first one in the list by default.")
    best_epoch_ind = best_epoch_inds[0]

    best_epoch = epochs[best_epoch_ind].long().item()
    acc_best_epoch = acc[best_epoch_ind].item()
    ece_best_epoch = ece[best_epoch_ind].item()

    print("Top",topk_perc,"Best epoch",best_epoch,acc_best_epoch,ece_best_epoch)

    return best_epoch

if __name__ == "__main__":

    path = "../results/CROHN25/metrics_noneRed_focal_val.csv"
    arr = np.genfromtxt(path,delimiter=",")[1:]

    arr = torch.from_numpy(arr)
    acc = arr[:,1]
    ece = arr[:,7]
    epochs = arr[:,0]

    #Simulate early stopping 
    max_worse_epoch = 5
    worse_epoch_nb = 0
    best_acc = 0
    i=0
    while i < epochs.max() and worse_epoch_nb <= max_worse_epoch:
        curr_acc = acc[i]
        if acc[i] < best_acc:
            worse_epoch_nb += 1 
        else:
            best_acc = acc[i]
            worse_epoch_nb = 0 
        i += 1 
    end_ind = i

    end_epoch = epochs[end_ind]
    print("End_epoch",end_epoch.item())

    arr = arr[:end_ind+1]

    acc = arr[:,1]
    ece = arr[:,7]
    epochs = arr[:,0]

    res = _acc_and_ece_selection(acc,ece,epochs)

    acc_and_ece_selection(path,topk_perc=0.1)