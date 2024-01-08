import torch
from saliency_maps_metrics.multi_step_metrics import Deletion, Insertion
from saliency_maps_metrics.single_step_metrics import IIC_AD, ADD
from utils import get_img_norm_func

metric_dic = {"DAUC":Deletion, "IAUC":Insertion, "AD":IIC_AD, "ADD":ADD}
default_metric_list = list(metric_dic.keys())
is_multi_step = {"DAUC":True, "IAUC":True, "AD":False, "ADD":False}
is_masking_object = {"DAUC":True,"IAUC":False,"AD":False,"ADD":True}

def get_att_maps(retDict):
  
    if not "attMaps" in retDict:
        attMaps = torch.abs(retDict["feat"].sum(dim=1,keepdim=True))
    else:
        attMaps = retDict["attMaps"]

    return attMaps

def apply_sal_metr_masks(data_orig,retDict=None,mask_prob=1,model=None,metric_list=None,adv_training=False,dataset_root=None):

    assert (retDict is not None) or (model is not None)
    
    if adv_training:
        data = get_img_norm_func(dataset_root,inverse=True)(data_orig)
    else:
        data = data_orig

    if metric_list is None:
        metric_list = default_metric_list

    if retDict is None:
        retDict = model(data)

    metric_constr_kwargs = {}

    data_masked_list = []
    expl = get_att_maps(retDict)
    is_masking_object_list = []
    mask_list = []
    is_iauc = torch.zeros(len(data)).to(data)
    for i in range(len(data)):
        if torch.rand(size=(1,)).item() <= mask_prob:
            metric_ind = torch.randint(0,len(metric_list),size=(1,)).item()
            metric_name = metric_list[metric_ind]
            is_iauc[i] = 1*(metric_name=="IAUC")
            is_masking_object_list.append(is_masking_object[metric_name])
            if is_multi_step[metric_name]:
                metric = metric_dic[metric_name](cumulative=True,**metric_constr_kwargs)

                data_i = data[i:i+1]
                masking_data_i = metric.get_masking_data(data_i)

                expl_i = expl[i:i+1]

                dic = metric.choose_data_order(data_i,masking_data_i)
                data1,data2 = dic["data1"],dic["data2"]

                k = torch.randint(0,expl.shape[2]*expl.shape[3],size=(1,)).item()

                total_pixel_nb = expl.shape[2]*expl.shape[3]
                step_nb = min(metric.max_step_nb,total_pixel_nb) if metric.bound_max_step else total_pixel_nb
                pixel_removed_per_step = total_pixel_nb//step_nb

                mask,_ = metric.compute_mask(expl_i,data.shape,k,pixel_removed_per_step)
                data_masked = metric.apply_mask(data1,data2,mask)
                data_masked_list.append(data_masked)
            else:
                metric = metric_dic[metric_name](**metric_constr_kwargs)

                data_i = data[i:i+1]

                masking_data_i = metric.get_masking_data(data_i)

                mask = metric.compute_mask(expl[i:i+1],data.shape)
                data_masked = metric.apply_mask(data_i,masking_data_i,mask)
                data_masked_list.append(data_masked)
            
            mask_list.append(mask)
        else:
            data_masked_list.append(data[i:i+1])
            is_masking_object_list.append(False)

    data_masked = torch.cat(data_masked_list,dim=0).to(data.device)
    
    if adv_training:
        data_masked = (data_masked-data_masked.min())/(data_masked.max()-data_masked.min())
        data_masked = get_img_norm_func(dataset_root)(data_masked)

    return data_masked,is_masking_object_list
        
def apply_sal_metr_masks_and_update_dic(model,data,args,resDict):

    data_masked,is_object_masked_list = apply_sal_metr_masks(data,resDict,args.sal_metr_mask_prob,adv_training=args.adv_ce_weight>0,dataset_root=args.dataset_train)
    resDict["is_object_masked_list"] = is_object_masked_list
    resDict_masked = model(data_masked) 
    resDict.update({key+"_masked":resDict_masked[key] for key in resDict_masked})
    return resDict,data,data_masked
        
    
        