import torch
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import os

def updateBestModel(metricVal,bestMetricVal,exp_id,model_id,bestEpoch,epoch,net,isBetter,worseEpochNb,output_dir):

    if isBetter(metricVal,bestMetricVal):
        best_path = f"{output_dir}/models/{exp_id}/model{model_id}_best_epoch{bestEpoch}"
        
        if os.path.exists(best_path):
            os.remove(best_path)

        torch.save(net.state_dict(), f"{output_dir}/models/{exp_id}/model{model_id}_best_epoch{epoch}")
        bestEpoch = epoch
        bestMetricVal = metricVal
        worseEpochNb = 0
    else:
        worseEpochNb += 1

    return bestEpoch,bestMetricVal,worseEpochNb

def all_cat_var_dic(var_dic,resDict,target):
    var_dic = cat_var_dic(var_dic,"output",resDict["output"])
    var_dic = cat_var_dic(var_dic,"output_masked",resDict["output_masked"])
    var_dic = cat_var_dic(var_dic,"target",target)
    return var_dic

def cat_var_dic(var_dic,tensor_name,tensor):
    
    assert tensor.ndim in [1,2,4]

    tensor = preproc_vect(tensor)

    if not tensor_name in var_dic:
        var_dic[tensor_name] = tensor
    else:
        var_dic[tensor_name] = torch.cat((var_dic[tensor_name],tensor),dim=0)

    return var_dic

def preproc_vect(vect):
    return vect.detach().cpu()