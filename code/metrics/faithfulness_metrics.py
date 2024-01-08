import matplotlib.pyplot as plt
plt.switch_backend('agg')
from saliency_maps_metrics.multi_step_metrics import Deletion, Insertion
from saliency_maps_metrics.single_step_metrics import IIC_AD, ADD

const_dic = {"Deletion":Deletion,"Insertion":Insertion,"IIC_AD":IIC_AD,"ADD":ADD}

def validity_rate_multi_step(metric_name,all_score_list):
    if metric_name == "Deletion":
        validity_rate = (all_score_list[:,:-1] > all_score_list[:,1:]).astype("float").mean()
    else: 
        validity_rate = (all_score_list[:,:-1] < all_score_list[:,1:]).astype("float").mean()
    return validity_rate

def validity_rate_single_step(metric_name,all_score_list,all_score_masked_list):
    if metric_name == "IIC_AD":
        validity_rate = (all_score_list < all_score_masked_list).astype("float").mean()
    else: 
        validity_rate = (all_score_list > all_score_masked_list).astype("float").mean()
    return validity_rate

def get_is_multi_step_dic():
    return {"Deletion":True,"Insertion":True,"IIC_AD":False,"ADD":False}

def get_metric_types(include_noncumulative=False):
    metric_types = const_dic.keys()
    if include_noncumulative:
        metric_types += ["Deletion-nc","Insertion-nc"]
    return metric_types

def get_ylim(metric):
    if metric in ["DC","IC","ADD"]:
        return (-1,1)
    else:
        return (0,1)

def get_correlation_metric_list():
    return ["DC","IC"]

def get_sub_multi_step_metric_list():
    return ["DAUC","DC","IAUC","IC"]

def get_sub_single_step_metric_list(include_iic=True):
    if include_iic:
        return ["AD","IIC","ADD"]
    else:
        return ["AD","ADD"]

def get_metrics_to_minimize():
    return ["DAUC","AD"]

def get_cumulative_suff_list():
    return ["","-nc"]

def get_sub_metric_list(include_iic=False,include_noncumulative=True,include_useless_metrics=False):
    metric_list = []

    metric_list.extend(get_sub_single_step_metric_list(include_iic))

    metric_list.extend(get_sub_cumulative_metric_list())

    if include_noncumulative:
        metric_list.extend(get_sub_noncumulative_metric_list(include_useless_metrics))

    return metric_list

def get_sub_cumulative_metric_list():
    return ["DAUC","IAUC","DC","IC"]

def get_sub_noncumulative_metric_list(include_useless_metrics=False):
    if include_useless_metrics:
        return ["DAUC-nc","IAUC-nc","DC-nc","IC-nc"]
    else:
        return ["DC-nc","IC-nc"]

def get_sal_metric_dics():
    return get_is_multi_step_dic(),const_dic
