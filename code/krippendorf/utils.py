import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from metrics.faithfulness_metrics import get_sub_metric_list,get_sub_noncumulative_metric_list
from post_hoc_expl.utils import get_all_group_names

def order_groups(groups,*matrices):
    group_correct_order = get_all_group_names(include_recent=True,include_all=True)
    groups_idx = [groups.index(group) for group in group_correct_order]
    matrices = order(matrices,groups_idx)
    return group_correct_order,matrices

def order_metrics(metrics,*matrices):
    metrics_correct_order = get_sub_metric_list()
    metrics_idx = [metrics.index(metric) for metric in metrics_correct_order]
    matrices = order(matrices,metrics_idx)
    return metrics_correct_order,matrices

def order(matrices,idx):
    re_ordered_matrices = []
    for matrix in matrices:
        if not matrix is None:
            matrix = np.array(matrix)[:,idx]
        re_ordered_matrices.append(matrix)
    return re_ordered_matrices

def get_aggr_kripp_dic(exp_id,model_id,background,post_hoc_group,metrics_to_remove,output_dir):
    filename_suff = f"{model_id}_b{background}"
    csv_path = f"{output_dir}/results/{exp_id}/krippendorff_alpha_{filename_suff}_{post_hoc_group}.csv"

    csv = np.genfromtxt(csv_path, delimiter=";",dtype=str)

    cumulative_metric_names = get_sub_metric_list(include_noncumulative=True)
    non_cumulative_metric_names = get_sub_noncumulative_metric_list(include_useless_metrics=True)

    interv_dic = {}
    for i in range(1,len(csv)):
        row = csv[i].split(",")
        cumulative = row[0] == "True"
 
        if cumulative:
            metric_names = cumulative_metric_names
        else:
            metric_names = non_cumulative_metric_names

        for j in range(len(row)-1):       
            mean_low_high = row[j+1]
            mean_low_high = mean_low_high.replace("(","").replace(")","")
            mean,lower,upper = mean_low_high.split(" ")

            metric_name = metric_names[j]
            if not metric_name in metrics_to_remove:
                interv_dic[metric_name] = {"mean":float(mean),"lower":float(lower),"upper":float(upper)}

    return interv_dic

def interval_metric(a, b):
    return (a-b)**2

def ratio_metric(a,b):

    if a.dtype == "bool":
        result = (a==b)*1.0
    else:
        result = (((a-b)/(a+b))**2)
        result[a+b == 0] = 0

    return result

def krippendorff_alpha_bootstrap(*data,**kwargs):

    data = np.stack(data)
    
    if len(data.shape) == 3:
        data = data.transpose(1,0,2)
    else:
        data = data[np.newaxis]

    res_list = []

    for i in range(len(data)):
        res_list.append(krippendorff_alpha_paralel(data[i],**kwargs))
  
    return res_list

def make_n_dict(data):

    unit_nb = data.shape[1]

    o = np.zeros((unit_nb,unit_nb))

    for i in range(unit_nb):
        for j in range(unit_nb):
            o[i,j] = 0
            for u in range(unit_nb):
                number_of_ij_pairs = (data[:,u] == i+1).sum()*(data[:,u] == j+1).sum()
                o[i,j] += number_of_ij_pairs/(unit_nb-1)

    n_vector = o.sum(axis=1)

    diff_mat = np.zeros((unit_nb,unit_nb))

    for i in range(unit_nb):
        for j in range(unit_nb):
            if i >= j:
                start,end = j,i 
            else:
                start,end = i,j

            diff_mat[i,j] = sum([n_vector[k] for k in range(start,end+1)])
            diff_mat[i,j] -= (n_vector[i] + n_vector[j])/2

    diff_mat = np.power(diff_mat,2)

    return diff_mat

def ordinal_metric(a,b,diff_mat):

    shape = np.broadcast(a,b).shape

    a = np.broadcast_to(a,shape)
    b = np.broadcast_to(b,shape)

    diff = diff_mat[a.reshape(-1)-1,b.reshape(-1)-1]

    diff = diff.reshape(shape)

    return diff

def binary_metric(a,b):
    return (a==b)*1.0

metric_dict = {"ratio_metric":ratio_metric,"interval_metric":interval_metric,"ordinal_metric":ordinal_metric,"binary_metric":binary_metric}

#From https://github.com/grrrr/krippendorff-alpha/blob/master/krippendorff_alpha.py
def krippendorff_alpha_paralel(data, metric=ratio_metric, missing_items=None,axis=None):
    '''
    Calculate Krippendorff's alpha (inter-rater reliability):
    
    data is in the format
    [
        {unit1:value, unit2:value, ...},  # coder 1
        {unit1:value, unit3:value, ...},   # coder 2
        ...                            # more coders
    ]
    or 
    it is a sequence of (masked) sequences (list, numpy.array, numpy.ma.array, e.g.) with rows corresponding to coders and columns to items
    
    metric: function calculating the pairwise distance
    force_vecmath: force vector math for custom metrics (numpy required)
    convert_items: function for the type conversion of items (default: float)
    missing_items: indicator for missing items (default: None)
    '''

    if data.dtype == np.int64:
        metric= "ordinal_metric"
    elif data.dtype == bool:
        metric = "binary_metric"
        data = data.astype("int")
    else:
        raise ValueError("Unkown data type:",data.dtype)

    if type(metric) is str:
        metric = metric_dict[metric]

    if metric is ordinal_metric:
        
        diff_mat=make_n_dict(data)

        metric = lambda a,b:ordinal_metric(a,b,diff_mat)


    # number of coders
    m = len(data)
    
    # set of constants identifying missing values
    if missing_items is None:
        maskitems = []
    else:
        maskitems = list(missing_items)
    if np is not None:
        maskitems.append(np.ma.masked_singleton)
    
    # convert input data to a dict of items
    units = {}
    units = {j:data[:,j] for j in range(data.shape[1])}
    units = dict((it, d) for it, d in units.items() if len(d) > 1)  # units with pairable values

    n = data.size
  
    Do = 0.

    data_perm = data.transpose(1,0)

    Do = metric(data_perm[:,:,np.newaxis],data_perm[:,np.newaxis,:]).sum()  
    Do /= (n*(data.shape[0]-1))

    if Do == 0:
        return 1.

    De = metric(data_perm[np.newaxis,:,:,np.newaxis],data_perm[:,np.newaxis,np.newaxis,:]).sum()
  
    De /= float(n*(n-1))

    coeff = 1.-Do/De if (Do and De) else 1.

    return coeff

def krippendorff_alpha(data, metric=interval_metric, convert_items=float, missing_items=None,axis=None):
    '''
    Calculate Krippendorff's alpha (inter-rater reliability):
    
    data is in the format
    [
        {unit1:value, unit2:value, ...},  # coder 1
        {unit1:value, unit3:value, ...},   # coder 2
        ...                            # more coders
    ]
    or 
    it is a sequence of (masked) sequences (list, numpy.array, numpy.ma.array, e.g.) with rows corresponding to coders and columns to items
    
    metric: function calculating the pairwise distance
    force_vecmath: force vector math for custom metrics (numpy required)
    convert_items: function for the type conversion of items (default: float)
    missing_items: indicator for missing items (default: None)
    '''
    
    # number of coders
    m = len(data)
    
    # set of constants identifying missing values
    if missing_items is None:
        maskitems = []
    else:
        maskitems = list(missing_items)
    if np is not None:
        maskitems.append(np.ma.masked_singleton)
    
    # convert input data to a dict of items
    units = {}

    for d in data:
        try:
            # try if d behaves as a dict
            diter = d.items()
        except AttributeError:
            # sequence assumed for d
            diter = enumerate(d)
            
        for it, g in diter:
            if g not in maskitems:
                try:
                    its = units[it]
                except KeyError:
                    its = []
                    units[it] = its
                its.append(convert_items(g))

    units = dict((it, d) for it, d in units.items() if len(d) > 1)  # units with pairable values
    n = sum(len(pv) for pv in units.values())  # number of pairable values
    
    if n == 0:
        raise ValueError("No items to compare.")
        
    Do = 0.
    for grades in units.values():
        gr = np.asarray(grades)
        Du = sum(np.sum(metric(gr, gri)) for gri in gr)
        Do += Du/float(len(grades)-1)
    Do /= float(n)

    if Do == 0:
        return 1.

    De = 0.
    for g1 in units.values():
        d1 = np.asarray(g1)
        for g2 in units.values():
            De += sum(np.sum(metric(d1, gj)) for gj in g2)
    De /= float(n*(n-1))

    return 1.-Do/De if (Do and De) else 1.
