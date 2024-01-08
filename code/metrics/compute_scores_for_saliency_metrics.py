import os,sys
import glob
import hashlib

import numpy as np
import torch
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from args import ArgReader,str2bool,set_debug_mode
from model.init_model import preprocessAndLoadParams
from model.modelBuilder import get_feat_map_size,netBuilder
from metrics.faithfulness_metrics import get_sal_metric_dics
from data.load_data import getBatch,sample_img_inds,buildTestLoader
from utils import normalize_tensor,getExplanations
from post_hoc_expl.utils import getAttMetrMod

def find_class_first_image_inds(label_list: list) -> list:
    class_first_image_inds = [0]
    labels = [0]
    for i in range(len(label_list)):
        label = label_list[i]
        if label not in labels:
            labels.append(label)
            class_first_image_inds.append(i)
    
    return class_first_image_inds

def set_metric_step_nb(args: ArgReader) -> int:
    if args.multi_step_metrics_step_nb == "auto":
        step_nb = get_feat_map_size(args)**2
    else:
        step_nb = int(args.multi_step_metrics_step_nb)
    return step_nb

def loadSalMaps(exp_id: str,model_id: str,output_dir) -> torch.tensor:

    if not "transf" in model_id:
        norm_paths = glob.glob(f"{output_dir}/results/{exp_id}/norm_{model_id}_epoch*.npy")
   
        if len(norm_paths) != 1:
            raise ValueError(f"Wrong norm path number for exp {exp_id} model {model_id}")
        else:
            norm = torch.tensor(np.load(norm_paths[0],mmap_mode="r"))/255.
    else:
        norm_paths = []
        norm = torch.ones((1,1,1,1))
    
    if len(norm_paths) == 0:
        attMaps_paths = glob.glob(f"{output_dir}/results/{exp_id}/attMaps_{model_id}_epoch*.npy")
    else:
        attMaps_paths = glob.glob(norm_paths[0].replace("norm","attMaps"))
        
    if len(attMaps_paths) >1:
        raise ValueError(f"Wrong path number for exp {exp_id} model {model_id}")
    elif len(attMaps_paths) == 0:
        print(f"No attMaps found for {model_id}. Only using the norm.")
        salMaps = norm
    else:
        attMaps = torch.tensor(np.load(attMaps_paths[0],mmap_mode="r"))/255.0
        salMaps = attMaps*norm
        salMaps = normalize_tensor(salMaps,dim=(1,2,3))
        
    return salMaps

def get_attr_func(post_hoc_method,net,args):
    if args.perturbation_maps_resolution == "auto":
        resolution = get_feat_map_size(args)
    else:
        resolution = int(args.perturbation_maps_resolution)

    attrFunc,kwargs = getAttMetrMod(post_hoc_method,net,resolution,args)
    return attrFunc,kwargs

def get_hash(inds) -> str:
    inds_string = "-".join([str(ind) for ind in inds])
    hashed_inds = hashlib.sha1(inds_string.encode("utf-8")).hexdigest()[:16]
    return hashed_inds 

def get_tensor_path(key,exp_id:str,model_id:str,hashed_inds:str,post_hoc_method:str=None,output_dir="../") -> str:
    if key == "explanations":
        return f"{output_dir}/results/{exp_id}/explanations_{model_id}_{post_hoc_method}_{hashed_inds}.th"
    elif key == "output":
        return f"{output_dir}/results/{exp_id}/output_{model_id}_{hashed_inds}.th"
    else:
        raise ValueError(f"Wrong key {key}")
    
def load_output(inds,exp_id,model_id,output_dir):
    hashed_inds = get_hash(inds)
    output_path = get_tensor_path("output",exp_id,model_id,hashed_inds,output_dir=output_dir)
    output = torch.load(output_path)
    return output

def load_explanations(inds,exp_id,model_id,post_hoc_method,output_dir) -> torch.tensor:
    hashed_inds = get_hash(inds)
    expl_path = get_tensor_path("explanations",exp_id,model_id,post_hoc_method,hashed_inds,output_dir=output_dir)
    explanations = torch.load(expl_path)
    return explanations

def compute_or_load_explanations(inds,exp_id:str,model_id:str,post_hoc_method:str,data:torch.tensor,predClassInds:torch.tensor,attrFunc,kwargs:dict,args:ArgReader) -> torch.tensor:
    hashed_inds = get_hash(inds)

    expl_path = get_tensor_path("explanations",exp_id,model_id,post_hoc_method,hashed_inds,output_dir=args.output_dir)
    print(expl_path)
    if not os.path.exists(expl_path):
        print("Computing explanations")
        explanations = getExplanations(post_hoc_method,inds,data,predClassInds,attrFunc,kwargs,args)
        torch.save(explanations.cpu(),expl_path)
    else:
        print("Already computed explanations")
        explanations = torch.load(expl_path).to(data.device)
    
    if args.perturbation_maps_resolution == "auto":
        resolution = get_feat_map_size(args)
    else:
        resolution = int(args.perturbation_maps_resolution)
    
    explanations = torch.nn.functional.interpolate(explanations,resolution)

    return explanations

def compute_or_load_output(inds,exp_id,model_id,data,net_lambda,output_dir):
    hashed_inds = get_hash(inds)
    output_path = get_tensor_path("output",exp_id,model_id,hashed_inds,output_dir=output_dir)
    if not os.path.exists(output_path):
        print("Computing output")
        output = net_lambda(data)
        torch.save(output.cpu(),output_path)
    else:
        print("Already computed output")
        output = torch.load(output_path).to(data.device)
    return output


def get_data_and_model(model_id,args :ArgReader,return_attr_func=True,set_net_as_lambda=True,post_hoc_method=None,output_dir="../") -> tuple:
    _,testDataset = buildTestLoader(args, "test")

    path = glob.glob(f"{output_dir}/models/{args.exp_id}/model{model_id}_best_epoch*")[0]

    net_original = netBuilder(args)
    net_original = preprocessAndLoadParams(path,args.cuda,net_original,verbose=False)
    net_original.eval()
    if set_net_as_lambda:
        net = lambda x:net_original(x)["output"]
    else:
        net = net_original

    inds = sample_img_inds(args.img_nb_per_class,testDataset=testDataset)
    data,target = getBatch(testDataset,inds,args)
    
    if return_attr_func:
        attrFunc,kwargs = get_attr_func(post_hoc_method,net_original,args)
        return data,target,inds,testDataset,attrFunc,kwargs,net
    else:
        return data,target,inds,testDataset,net

def add_compute_scores_args(argreader):
    argreader.parser.add_argument('--attention_metric', type=str, help='The attention metric to compute.')
    argreader.parser.add_argument('--cumulative', type=str2bool, help='To prevent acumulation of perturbation when computing metrics.',default=True)
    return argreader

def compute_scores(attention_metric,post_hoc_method,model_id,cumulative,args):

    args = set_debug_mode(args)
    args.cuda = args.cuda and torch.cuda.is_available()

    args.multi_step_metrics_step_nb = set_metric_step_nb(args)

    #Constructing result file path
    post_hoc_suff = "" if post_hoc_method is None else "-"+post_hoc_method
    formated_attention_metric = attention_metric.replace("_","")

    #Constructing metric
    is_multi_step_dic,const_dic = get_sal_metric_dics()
    metric_constr_arg_dict = {}

    if is_multi_step_dic[attention_metric]:
        metric_constr_arg_dict.update({"cumulative":cumulative})

        if not cumulative:
            formated_attention_metric += "nc"

    metric = const_dic[attention_metric](**metric_constr_arg_dict)

    data_replace_method = metric.data_replace_method 
 
    result_file_path = f"{args.output_dir}/results/{args.exp_id}/{formated_attention_metric}-{data_replace_method}_{model_id}{post_hoc_suff}.npy"
    
    print(result_file_path)
    result_file_exists = os.path.exists(result_file_path)

    if not result_file_exists:
        
        if args.debug:
            args.rise_mask_nb = 2
            args.nt_samples = 2

        data,_,inds,_,attrFunc,kwargs,net_lambda = get_data_and_model(model_id,args,post_hoc_method=post_hoc_method,output_dir=args.output_dir)

        torch.set_grad_enabled(False)

        outputs = compute_or_load_output(inds,args.exp_id,model_id,data,net_lambda,output_dir=args.output_dir)
        predClassInds = outputs.argmax(dim=-1)

        if post_hoc_method == "gradcam_pp":
            torch.set_grad_enabled(True)

        if args.debug:
            data = data[:args.debug_batch_size]
            inds = inds[:args.debug_batch_size]
            predClassInds = predClassInds[:args.debug_batch_size]

        explanations = compute_or_load_explanations(inds,args.exp_id,model_id,post_hoc_method,data,predClassInds,attrFunc,kwargs,args)

        torch.set_grad_enabled(False)   

        metric_args = [net_lambda,data,explanations.float(),predClassInds]
        kwargs = {"save_all_class_scores":True}

        if is_multi_step_dic[attention_metric]:  
            scores,saliency_scores = metric.compute_scores(*metric_args,**kwargs)
            dic_to_save = {"prediction_scores":scores,"saliency_scores":saliency_scores}
        else:
            scores,scores_masked = metric.compute_scores(*metric_args,**kwargs)
            dic_to_save = {"prediction_scores":scores,"prediction_scores_with_mask":scores_masked}
        
        np.save(result_file_path,dic_to_save)
    else:
        data,_,inds,_,net_lambda = get_data_and_model(model_id,args,return_attr_func=False,output_dir=args.output_dir)
        hashed_inds = get_hash(inds)
        output_path = get_tensor_path("output",args.exp_id,args.model_id,hashed_inds)    
        if not os.path.exists(output_path): 
            torch.set_grad_enabled(False)
            output = net_lambda(data)
            torch.save(output.cpu(),output_path)
        else:
            print("Already done")
        