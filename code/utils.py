
import os,glob,json
import collections

import numpy as np
import torchvision
from torchvision import transforms 
import torch 
import matplotlib.pyplot as plt 
import matplotlib

from post_hoc_expl.utils import applyPostHoc
from data.fineGrainedDataset import get_imgnet_mean_std
from args import get_arg_from_model_config_file

def plot_difference_matrix_as_figure(matrix,baseline_matrix,labels1,labels2,filename,is_significant_matrix=None,output_dir="../",min_diff=-50,max_diff=50,fontsize=13,print_mean_and_baseline_values=True,scale=8,aspect_ratio=None,make_max_bold=False,color_bar=True,inv_diff=False):
    
    if not isinstance(matrix,(np.ndarray, np.generic)):
        baseline_matrix = np.array(baseline_matrix)
        matrix = np.array(matrix)
    
    if inv_diff:
        difference_matrix = baseline_matrix - matrix 
    else:
        difference_matrix = matrix - baseline_matrix
    difference_matrix_norm = (difference_matrix - min_diff)/(max_diff-min_diff)

    #plot the difference matrix as a heat map from red to green
    #If the difference is not significant, the cell is a small square
    #If the difference is significant, the cell is a large square
    if aspect_ratio is None: 
        aspect_ratio = len(labels1)/len(labels2)
    fig, ax = plt.subplots(figsize=(scale,scale*aspect_ratio))
    ax.imshow(np.zeros_like(difference_matrix),cmap="Greys",vmin=0,vmax=100,aspect=aspect_ratio)

    #Red to green cmap 
    cmap = matplotlib.colormaps['RdYlGn']

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

    #plt.imshow(p_val_mat*0,cmap="Greys")

    if color_bar:
        #Add a RdYlGn colorbar
        norm = matplotlib.colors.Normalize(vmin=min_diff, vmax=max_diff)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm,ax=ax)
        cbar.ax.tick_params(labelsize=fontsize)

    if matrix.dtype == "int64":
        fmt_func = lambda x:x
    else:
        fmt_func = lambda x:f"{x:.1f}"

    if make_max_bold:
        max_values = difference_matrix.max(axis=0)

    #Ad the difference value at each cell of the matrix
    for i in range(len(labels2)):
        for j in range(len(labels1)):

            #Set text color appropriately depending on the value of the difference
            #If the absolute value of the difference is greater than 10, the text is white
            #Otherwise, the text is black
            #color = "white" if abs(difference_matrix[i,j]) > 40 else "black"
            color = "black"

            if is_significant_matrix is not None:
                size = 1 if is_significant_matrix[i,j] else 0.6
            else:
                size = 1 

            circle = matplotlib.patches.Rectangle((j-size/2, i-size/2), width=size,height=size, color=cmap(difference_matrix_norm[i,j]))
            ax.add_patch(circle)
            
            if make_max_bold and difference_matrix[i,j] == max_values[j]:
                fontweight = "bold"
            else:
                fontweight = "regular"

            if not print_mean_and_baseline_values:
                text = ax.text(j, i, fmt_func(difference_matrix[i, j]),ha="center", va="center", color=color,fontsize=fontsize,fontweight=fontweight)

            else:
                text = ax.text(j, i-0.1, fmt_func(difference_matrix[i, j]),ha="center", va="center", color=color,fontsize=fontsize,fontweight=fontweight)
            
                baseline_and_new_mean = f"{fmt_func(baseline_matrix[i,j])} → {fmt_func(matrix[i,j])}"
            
                text = ax.text(j, i+0.3, baseline_and_new_mean,ha="center", va="center", color=color,fontsize=fontsize*0.6,alpha=0.7,fontweight=fontweight)
            
    #Ad the dataset name 
    ax.set_xticks(np.arange(len(labels1)))
    ax.set_yticks(np.arange(len(labels2)))
    if not print_mean_and_baseline_values:
        ax.set_xticklabels(labels1,fontsize=fontsize)
        ax.set_yticklabels(labels2,fontsize=fontsize)
    else:
        ax.set_xticklabels(labels1,fontsize=fontsize)
        ax.set_yticklabels(labels2,fontsize=fontsize)       

    #Save the figure 
    fig.tight_layout()
    plt.savefig(output_dir+"/vis/"+filename)
    plt.close() 

class Tree(collections.defaultdict):
    def __init__(self, value=None):
        super(Tree, self).__init__(Tree)
        self.value = value

def get_post_hoc_group_label_dict():
    return {"perturb":"Perturbation","recent":"Recent","all":"All","classmap":"Class-Map","backprop":"BP"}

def get_post_hoc_label_dic():
    return {"randomfeatmap":"RandFeatMap","topfeatmap":"TopFeatMap","randommap":"RandomMap","cam":"CAM","am":"AM","ablationcam":"Ablation-CAM","gradcam":"Grad-CAM","gradcampp":"Grad-CAM++","scorecam":"Score-CAM","rise":"RISE","varGrad":"VarGrad","smoothGrad":"SmoothGrad","intGrad":"IntGrad","guided":"GuidedBackprop","featurepermutation":"FeatPerm","featureablation":"FeatAbl","occlusion":"Occlusion"}

def get_dataset_labels(config_paths):

    dataset_label_dic = get_dataset_label_dic()
    labels = []

    for config_path in config_paths:
        dataset_test = get_arg_from_model_config_file("dataset_test",config_path)
        dataset_test = dataset_test.replace("_test","")
        labels.append(dataset_label_dic[dataset_test])

    return labels

def get_dataset_label_dic():
    return {"CUB_200_2011":"CUB","cars":"Standford cars","aircraft":"FGVC","embryo_img":"Embryo","embryo_img_fixed":"Embryo","crohn_2":"Crohn"}

def get_configs(config_paths=None,config_fold="configs/"):

    if config_paths is None:
        config_paths = glob.glob(os.path.join(config_fold,"*.config"))
        config_paths = list(filter(lambda x:"debug" not in os.path.basename(x),config_paths))

    return config_paths

def get_model_ids(model_ids,model_args_path,only_known_model_ids=False):

    with open(model_args_path) as json_file:
        known_model_ids = json.load(json_file).keys()

    if model_ids is None:
        model_ids = known_model_ids
    else:
        if only_known_model_ids:
            assert all([model_id in known_model_ids for model_id in model_ids]),"At least one model id is unkown. Please only enter model ids defined in model dictionary in model_args.json."

    return model_ids

def get_img_norm_func(root,inverse=False):
    if "emb" not in root:
        return get_imgnet_norm(inverse=inverse)
    else:
        return lambda x:x
                                    
def get_img_norm_params(root):
    if "emb" not in root:
        return get_imgnet_mean_std()
    else:
        return {"mean":[0],"std":[1]}

def get_imgnet_norm(inverse=False):

    mean_and_std_dict = get_imgnet_mean_std()
    mean,std = mean_and_std_dict["mean"],mean_and_std_dict["std"]

    if inverse:
        transf= transforms.Compose([transforms.Normalize(mean = [ 0., 0., 0. ],std =[ 1./s for s in std]),
                                    transforms.Normalize(mean = [ -m for m in mean],std = [ 1., 1., 1. ])])
    else:
        transf = transforms.Normalize(mean=mean, std=std)

    return transf

def make_grid(img,row_nb):
    assert len(img) % row_nb == 0
    col_nb = len(img)//row_nb 
    img = img.reshape(row_nb,col_nb,img.shape[1],img.shape[2],img.shape[3])

    grid = []
    for i in range(row_nb):
        row = []
        for j in range(col_nb):
            row.append(img[i,j])
        
        row = torch.cat(row,dim=2)
        grid.append(row)

    grid = torch.cat(grid,dim=1)
    grid = grid.unsqueeze(0)

    return grid

def save_image(img,path,mask=None,apply_inv_norm=True,row_nb=None,**kwargs):
    
    if img.shape[1] == 3:
        if mask is None:
            mask = (img!=0)
        if apply_inv_norm:
            inv_imgnet_norm = get_imgnet_norm(inverse=True)
            img = inv_imgnet_norm(img)*mask
    
    if not row_nb is None:
        img = make_grid(img,row_nb)

    torchvision.utils.save_image(img,path,**kwargs)

class Bunch(object):
    def __init__(self, adict):
        self.__dict__.update(adict)

def normalize_tensor(tensor,dim=None):

    if dim is None:
        tensor = (tensor-tensor.min())/(tensor.max()-tensor.min())
    else:
        tensor_min = tensor
        tensor_max = tensor

        if type(tensor) is torch.Tensor:
            for i in dim:
                tensor_min = tensor_min.min(dim=i,keepdim=True)[0]
                tensor_max = tensor_max.max(dim=i,keepdim=True)[0]
        else:
            for i in range(len(dim)):
                tensor_min = tensor_min.min(axis=i,keepdims=True)[0]
                tensor_max = tensor_max.max(axis=i,keepdims=True)[0]
        
        tensor = (tensor-tensor_min)/(tensor_max-tensor_min)

    return tensor

def findNumbers(x):
    '''Extracts the numbers of a string and returns them as an integer'''

    return int((''.join(xi for xi in str(x) if xi.isdigit())))

def getEpoch(path):
    return int(os.path.basename(path).split("epoch")[1].split("_")[0])

def findLastNumbers(weightFileName):
    '''Extract the epoch number of a weith file name.

    Extract the epoch number in a weight file which name will be like : "clustDetectNet2_epoch45".
    If this string if fed in this function, it will return the integer 45.

    Args:
        weightFileName (string): the weight file name
    Returns: the epoch number

    '''

    i=0
    res = ""
    allSeqFound = False
    while i<len(weightFileName) and not allSeqFound:
        if not weightFileName[len(weightFileName)-i-1].isdigit():
            allSeqFound = True
        else:
            res += weightFileName[len(weightFileName)-i-1]
        i+=1

    res = res[::-1]

    return int(res)

def getExplanations(post_hoc_method,inds,data,predClassInds,attrFunc,kwargs,args):
    explanations = []

    bs = args.expl_batch_size
    batch_nb = len(data)//bs + 1*(len(data)%args.expl_batch_size>0)
    all_expl = []
    for i in range(batch_nb):
        ind,data_i,predClassInd = inds[i*bs:(i+1)*bs],data[i*bs:(i+1)*bs],predClassInds[i*bs:(i+1)*bs]
        if post_hoc_method:
            explanations = applyPostHoc(post_hoc_method,attrFunc,data_i,predClassInd,kwargs)
        else:
            explanations = attrFunc(ind)   
        all_expl.append(explanations)
    all_expl = torch.cat(all_expl,dim=0)

    return all_expl 