import matplotlib.pyplot as plt
plt.switch_backend('agg')
from metrics.calibration_metrics import ece,ada_ece,class_ece

def add_losses_to_dic(metDictSample,lossDic):
    for loss_name in lossDic:
        metDictSample[loss_name] = lossDic[loss_name].item()
    return metDictSample

def updateMetrDict(metrDict,metrDictSample):

    if metrDict is None:
        metrDict = metrDictSample
    else:
        for metric in metrDict.keys():
            metrDict[metric] += metrDictSample[metric]

    return metrDict

def binaryToMetrics(output,target,resDict):
    ''' Computes metrics over a batch of targets and predictions

    Args:
    - output (list): the batch of outputs
    - target (list): the batch of ground truth class
    - transition_matrix (torch.tensor) : this matrix contains at row i and column j the empirical probability to go from state i to j

    '''

    acc = compAccuracy(output,target)
    metDict = {"Accuracy":acc}

    for key in resDict.keys():
        if key.find("output_") != -1:
            suff = key.split("_")[-1]

            metDict["Accuracy_{}".format(suff)] = compAccuracy(resDict[key],target)

    return metDict
      
def compAccuracy(output,target):
    pred = output.argmax(dim=-1)
    acc = (pred == target).float().sum()
    return acc.item()

#From https://github.com/torrvision/focal_calibration/blob/main/Metrics/metrics.py
def expected_calibration_error(var_dic,metrDict):

    func_dic= {"ECE":ece,"AdaECE":ada_ece,"ClassECE":class_ece}

    for metric_name in func_dic:

        metrDict[metric_name] = func_dic[metric_name](var_dic["output"], var_dic["target"])

        if "output_masked" in var_dic:
            metrDict[metric_name+"_masked"] = func_dic[metric_name](var_dic["output_masked"], var_dic["target"])
    
    return metrDict