import os
import glob
import sqlite3

import numpy as np
import torch 
from saliency_maps_metrics.multi_step_metrics import compute_auc_metric

from args import set_debug_mode
from metrics.faithfulness_metrics import get_sal_metric_dics,validity_rate_multi_step,validity_rate_single_step
from .compute_scores_for_saliency_metrics import load_output
from data.load_data import buildTestLoader,sample_img_inds,getBatch
import utils

def add_output_if_missing(scores_dic,inds,exp_id,model_id,output_dir):
    if "outputs" not in scores_dic:
        output = load_output(inds,exp_id,model_id,output_dir)
        scores_dic["outputs"] = output
    return scores_dic

def get_db(exp_id,output_dir):
    db_path = f"{output_dir}/results/{exp_id}/saliency_metrics.db"
    if not os.path.exists(db_path):
        header = get_header()
        make_db(header,db_path)
    con = sqlite3.connect(db_path) # change to 'sqlite:///your_filename.db'
    cur = con.cursor()
    return con,cur 

def make_db(header,db_path):

    con = sqlite3.connect(db_path) # change to 'sqlite:///your_filename.db'
    cur = con.cursor()

    header = ",".join(header)

    unique_cols = get_unique_columns()
    unique_cols = ",".join(unique_cols)

    cur.execute(f"CREATE TABLE metrics ({header},UNIQUE({unique_cols}) ON CONFLICT IGNORE);")

    con.commit()

def apply_softmax(array,inds):
    tensor = torch.from_numpy(array)
    tensor = torch.softmax(tensor.double(),dim=-1)
    
    inds = inds.cpu()

    if len(tensor.shape) == 3:
        inds = inds.unsqueeze(-1).unsqueeze(-1)
        inds = inds.expand(-1,tensor.shape[1],-1)
        tensor = tensor.gather(2,inds).squeeze(-1)
    else:
        inds = inds.unsqueeze(-1)
        tensor = tensor.gather(1,inds).squeeze(-1)        

    array = tensor.numpy()
    return array

def fix_type(tensor):

    if type (tensor) is not np.ndarray:
        tensor = tensor.numpy()

    return tensor

def get_score_file_paths(exp_id,metric_list,pref="",output_dir="../"):
    paths = []
    for metric in metric_list:
        metric = metric.replace("_","")
        paths.extend(glob.glob(f"{output_dir}/results/{exp_id}/{pref}{metric}*_*.npy"))
    return paths

def get_supp_kw():
    return {5:"saliency_scores",6:"outputs",7:"target",8:"inds",9:"prediction_scores"}

def get_col_index_to_value_dic():
    dic = {0:"metric_label",1:"replace_method",2:"model_id",3:"post_hoc_method",4:"metric_value"}
    dic.update(get_supp_kw())
    return dic 

def write_db(cur,**kwargs):

    col_index_to_value_dic = get_col_index_to_value_dic()
    inds = sorted(list(col_index_to_value_dic.keys()))
    
    header = []
    value_list = []

    for ind in inds:
        if col_index_to_value_dic[ind] in kwargs:
            header.append(col_index_to_value_dic[ind])
            value_list.append(str(kwargs[col_index_to_value_dic[ind]]))

    header = ",".join(header)

    query = f'SELECT metric_value,inds FROM metrics WHERE model_id=="{kwargs["model_id"]}" and metric_label=="{kwargs["metric_label"]}" and replace_method=="{kwargs["replace_method"]}" and post_hoc_method=="{kwargs["post_hoc_method"]}"'

    query_res = cur.execute(query).fetchall()

    if len(query_res) > 0:
        metric_value,inds = zip(*query_res)

        consistent = True
        for key,value in zip(["metric_value","inds"],[metric_value[0],inds[0]]):
            if key in kwargs and value != str(kwargs[key]):
                print(f"Row already exists but with a different {key}. Old value: {value}, new value: {kwargs[key]}")
                print("The following kwargs were used",kwargs)
                consistent=False 
            if not consistent:
                raise ValueError("One or more inconsistent values have been found.")

    else:       
        question_marks = ",".join(['?' for _ in range(len(value_list))])
        cur.execute(f"INSERT INTO metrics ({header}) VALUES ({question_marks});", value_list)

def get_unique_columns():
    col_index_to_value_dic = get_col_index_to_value_dic()
    inds = sorted(list(col_index_to_value_dic.keys()))

    column_names = []

    for ind in inds:
        if col_index_to_value_dic[ind] in ["metric_label","replace_method","model_id","post_hoc_method"]:
            column_names.append(col_index_to_value_dic[ind])
    
    return column_names

def get_header():
    col_index_to_value_dic = get_col_index_to_value_dic()
    inds = sorted(list(col_index_to_value_dic.keys()))
    column_names = [col_index_to_value_dic[ind] for ind in inds]
    return column_names

def write_csv(**kwargs):

    col_index_to_value_dic = get_col_index_to_value_dic()

    inds = sorted(list(col_index_to_value_dic.keys()))
    value_list = [str(kwargs[col_index_to_value_dic[ind]]) for ind in inds]
    row = ",".join(value_list)
    exp_id = kwargs["exp_id"]
    output_dir =kwargs["output_dir"]

    csv_path = f"{output_dir}/results/{exp_id}/saliency_metrics.csv"

    if not os.path.exists(csv_path):
        column_names = [col_index_to_value_dic[ind] for ind in inds]
        header = ",".join(column_names)  
        with open(csv_path,"a") as file:
            print(header,file=file)

    with open(csv_path,"a") as file:
        print(row,file=file)

def list_to_fmt_str(array):
    if type(array) is torch.Tensor:
        array = array.cpu().numpy()
    if len(array.shape) > 1:
        fmt_str = f"shape={str(array.shape)};"
    else:
        fmt_str = ""
    fmt_str += str(";".join(array.reshape(-1).astype("str")))  
    return fmt_str

def get_info(path):

    filename = os.path.basename(path).replace(".npy","")
        
    underscore_ind = filename.find("_")
    metric_name_and_replace_method,model_id_and_posthoc_method = filename[:underscore_ind],filename[underscore_ind+1:]
    
    metric_name,replace_method = metric_name_and_replace_method.split("-")
    
    if metric_name == "IICAD":
        metric_name = "IIC_AD"

    kwargs = {}
    if "nc" in metric_name:
        metric_name = metric_name.replace("nc","")
        kwargs["cumulative"] = False
        suff = "-nc"
    else:
        suff = ""

    if "-" in model_id_and_posthoc_method:
        model_id,post_hoc_method = model_id_and_posthoc_method.split("-")
    else:
        model_id = model_id_and_posthoc_method
        post_hoc_method = ""
    
    return model_id,post_hoc_method,metric_name,replace_method,kwargs,suff

def compute_sal_metrics(exp_id,args):

    args = set_debug_mode(args)

    is_multi_step_dic,const_dic = get_sal_metric_dics()
    metric_list = list(const_dic.keys())
    score_file_paths = get_score_file_paths(exp_id,metric_list,output_dir=args.output_dir)
    
    con,cur = get_db(exp_id,args.output_dir)

    _,testDataset = buildTestLoader(args, "test")
    inds = sample_img_inds(args.img_nb_per_class,testDataset=testDataset)

    for path in score_file_paths:
        
        model_id,post_hoc_method,metric_name,replace_method,kwargs,suff = get_info(path)

        if args.model_ids is None or model_id in args.model_ids:

            metric = const_dic[metric_name](**kwargs)

            scores_dic = np.load(path,allow_pickle=True).item()
            scores_dic = add_output_if_missing(scores_dic,inds,exp_id,model_id,args.output_dir)

            if args.debug:
                for key in scores_dic:
                    scores_dic[key] = scores_dic[key][:args.debug_batch_size]

            if is_multi_step_dic[metric_name]:
                all_score_list,all_sal_score_list = scores_dic["prediction_scores"],scores_dic["saliency_scores"]
                all_score_list = fix_type(all_score_list)

                predClassInds = scores_dic["outputs"].argmax(dim=-1)
                
                if len(all_score_list.shape) == 3:
                    all_score_list = apply_softmax(all_score_list,predClassInds)

                auc_metric = compute_auc_metric(all_score_list)        
                calibration_metric = metric.compute_calibration_metric(all_score_list, all_sal_score_list)

                auc_metric = list_to_fmt_str(auc_metric)
                calibration_metric = list_to_fmt_str(calibration_metric)

                result_dic = metric.make_result_dic(auc_metric,calibration_metric)

                result_dic[metric_name+"_val_rate"] = validity_rate_multi_step(metric_name,all_score_list)

            else:
                all_score_list,all_score_masked_list = scores_dic["prediction_scores"],scores_dic["prediction_scores_with_mask"]
                all_score_list = fix_type(all_score_list)
                all_score_masked_list = fix_type(all_score_masked_list)

                predClassInds = scores_dic["outputs"].argmax(dim=-1)
                
                if len(all_score_list.shape) == 2:
                    all_score_list = apply_softmax(all_score_list,predClassInds)
                    all_score_masked_list = apply_softmax(all_score_masked_list,predClassInds)
 
                result_dic = metric.compute_metric(all_score_list,all_score_masked_list)
                
                for sub_metric in result_dic:
                    result_dic[sub_metric] = list_to_fmt_str(result_dic[sub_metric])

                result_dic[metric_name+"_val_rate"] = validity_rate_single_step(metric_name,all_score_list,all_score_masked_list)

            for sub_metric in result_dic.keys():
                write_db(cur,exp_id=exp_id,metric_label=sub_metric.upper()+suff,replace_method=replace_method,model_id=model_id,post_hoc_method=post_hoc_method,metric_value=result_dic[sub_metric],output_dir=args.output_dir)

    con.commit()
    con.close()