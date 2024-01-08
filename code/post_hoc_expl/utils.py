import numpy as np 
import torch
from captum.attr import (IntegratedGradients,NoiseTunnel,LayerGradCam,GuidedBackprop,Occlusion,FeaturePermutation,FeatureAblation)

from model import modelBuilder
from .scorecam import ScoreCam
from .gradcampp import LayerGradCampp
from .baselines import get_baseline_dict
from .rise import RISE 

def get_baseline_methods():
    return ["randommap","topfeatmap","randomfeatmap","am"]

def get_classmap_methods():
    return ["cam","gradcam","gradcampp","scorecam"]

def get_backprop_methods():
    return ["varGrad","smoothGrad","intGrad","guided"]

def get_perturb_methods():
    return ["featurepermutation","featureablation","occlusion","rise"]

def get_all_group_names(include_recent=False,include_all=False):
    
    groups = []
    if include_all:
        groups.append("all")

    groups += ["classmap","backprop","perturb"]
    
    if include_recent:
        groups.append("recent")

    return groups

def get_all_methods():
    all_methods = []
    all_groups = get_all_group_names()
    func_dic = globals()
    for group in all_groups:
        sub_list = func_dic["get_"+group+"_methods"]()
        all_methods.extend(sub_list)
    return all_methods

def get_recent_methods():
    return ["scorecam","rise","smoothGrad"]

def get_method_ind(method_name):
    all_methods = np.array(get_all_methods())
    return np.argwhere(all_methods==method_name)[0][0]

def applyPostHoc(post_hoc_method,attrFunc,data,targ,kwargs):

    argList = [data]
    baseline_dic = get_baseline_dict()
    if (post_hoc_method not in baseline_dic) or (post_hoc_method in baseline_dic and baseline_dic[post_hoc_method]["requ_targ"]):
        kwargs["target"] = targ

    attMap = attrFunc(*argList,**kwargs).clone().detach().to(data.device)

    if len(attMap.size()) == 2:
        attMap = attMap.unsqueeze(0).unsqueeze(0)
    elif len(attMap.size()) == 3:
        attMap = attMap.unsqueeze(0)
        
    return attMap

def postprocess_backprop_methods(func):
    return lambda *args,**kwargs:func(*args,**kwargs).mean(dim=1,keepdim=True)

def postprocess_perturb_methods(func,interp_size):
    return lambda *args,**kwargs:torch.nn.functional.interpolate(func(*args,**kwargs).mean(dim=1,keepdim=True),size=interp_size)

def get_feature_mask(img_size,resolution,cuda):
    arr = torch.arange(resolution*resolution)
    arr = arr.view(1,1,resolution,resolution)
    arr = arr.expand(-1,3,-1,-1)
    arr = torch.nn.functional.interpolate(arr.float(),size=img_size,mode="nearest").long()

    if cuda:
        arr = arr.cuda()

    return arr

def getAttMetrMod(post_hoc_method,net,resolution,args):

    baseline_dict = get_baseline_dict()

    print("Resolution is",resolution)

    if post_hoc_method in baseline_dict.keys():
        attrMod = baseline_dict[post_hoc_method]["const"](net)
        attrFunc = attrMod.forward
        kwargs = {}    
    elif post_hoc_method == "gradcam":
        netGradMod = modelBuilder.GradCamMod(net)
        attrMod = LayerGradCam(netGradMod.forward,netGradMod.layer4)
        attrFunc = attrMod.attribute
        kwargs = {}
    elif post_hoc_method == "gradcampp":
        netGradMod = modelBuilder.GradCamMod(net)
        attrMod = LayerGradCampp(netGradMod.forward,netGradMod.layer4)
        attrFunc = attrMod.attribute
        kwargs = {}
    elif post_hoc_method == "scorecam":
        attrMod = ScoreCam(net)
        attrFunc = attrMod.generate_cam
        kwargs = {}
    elif post_hoc_method == "guided":
        netGradMod = modelBuilder.GradCamMod(net)
        attrMod = GuidedBackprop(netGradMod)
        attrFunc = postprocess_backprop_methods(attrMod.attribute)
        kwargs = {}
    elif post_hoc_method in ["varGrad","smoothGrad","intGrad"]:
        args.expl_batch_size = 1
        net.eval()
        netGradMod = modelBuilder.GradCamMod(net)

        ig = IntegratedGradients(netGradMod)
        if post_hoc_method in ["varGrad","smoothGrad"]:
            attrMod = NoiseTunnel(ig)

            kwargs = {"nt_type":'smoothgrad_sq' if post_hoc_method == "smoothGrad" else "vargrad","stdevs":0.02, "nt_samples":args.nt_samples,"nt_samples_batch_size":args.noise_tunnel_batchsize}
        else:
            attrMod = ig
            kwargs = {}

        attrFunc = postprocess_backprop_methods(attrMod.attribute)
    
    elif post_hoc_method == "rise":
        torch.set_grad_enabled(False)
        attrMod = RISE(net,res=resolution,nbMasks=args.rise_mask_nb)
        attrFunc = attrMod.__call__
        kwargs = {}
    elif post_hoc_method == "occlusion":
        torch.set_grad_enabled(False)
        netGradMod = modelBuilder.GradCamMod(net)
        attrMod = Occlusion(netGradMod.forward)
        attrFunc = postprocess_perturb_methods(attrMod.attribute,resolution)
        window_size = args.img_size//resolution
        print("Occlusion window size",window_size)
        kwargs = {"sliding_window_shapes":(3,window_size,window_size), "strides":window_size}
    elif post_hoc_method == "featurepermutation":
        torch.set_grad_enabled(False)
        netGradMod = modelBuilder.GradCamMod(net)
        attrMod = FeaturePermutation(netGradMod.forward)
        attrFunc = postprocess_perturb_methods(attrMod.attribute,resolution)
        feature_mask = get_feature_mask(args.img_size,resolution,args.cuda)
        kwargs = {"feature_mask":feature_mask}
    elif post_hoc_method == "featureablation":
        torch.set_grad_enabled(False)
        netGradMod = modelBuilder.GradCamMod(net)
        attrMod = FeatureAblation(netGradMod.forward)
        attrFunc = postprocess_perturb_methods(attrMod.attribute,resolution)
        feature_mask = get_feature_mask(args.img_size,resolution,args.cuda)
        kwargs = {"feature_mask":feature_mask}
    else:
        raise ValueError("Unknown post-hoc method",post_hoc_method)
    return attrFunc,kwargs
