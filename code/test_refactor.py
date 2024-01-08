import glob,sys,os
import subprocess
import configparser

import numpy as np
import sqlite3 

from args import ArgReader
from metrics.faithfulness_metrics import get_sub_metric_list,get_sub_cumulative_metric_list
from post_hoc_expl.utils import get_all_methods
from krippendorf.krippendorf import get_background_func,fmt_value_str

#Don't remove that line. The Tree class is actually used when loading the critical N dict (which is in pickle format)
from test_size_reduction import Tree 

def csv_to_perf(path,metric="accuracy"):
    csv = np.genfromtxt(path,delimiter=",",dtype=str)
    header = csv[0]
    accuracy_col = np.argwhere(header==metric)
    return float(csv[-1,accuracy_col])

def make_csv_row(header,results):

    csv_row = ""
    for i,key in enumerate(header):
        csv_row += str(results[key])
        if i != len(header)-1:
            csv_row += ","
    return csv_row

def run_step(step_dict,step):
    step_args = step_dict[step]["args"]
    print(step_args)
    result = subprocess.run(step_args,check=False)
    assert result != 0,step+" throwed an error."

def get_commit():
    cmd = "git rev-parse --short HEAD"
    commit = subprocess.check_output(cmd.split(" "),text=True)
    commit = commit.replace("\n","")
    return commit

def init_result_file(config_path,step,keys_dic):        
    exp_id = get_exp_id(config_path)
    os.makedirs(f"../results/{exp_id}",exist_ok=True)
    result_file_path = f"../results/{exp_id}/tests_result_{step}.csv"
    header = ["debug","commit","model_id"]
    header += keys_dic[step]
    if not os.path.exists(result_file_path):
        with open(result_file_path,"w") as file:
            print(",".join(header),file=file)
    return result_file_path,header

def get_exp_id(config_path):
    config = configparser.ConfigParser()
    config.read(config_path)
    exp_id = config["default"]["exp_id"]   
    return exp_id 

def get_keys_dic():
    return {"training":get_training_keys(),"expl":get_expl_keys(),"krippen":get_krippen_keys(),"test_size_red":get_test_size_red_keys()}

def get_training_keys():
    return ["train_acc","val_acc","test_acc","train_cal","val_cal","test_cal"]

def get_expl_keys():
    keys = []
    metrics = get_sub_metric_list(True,True)
    explanations = get_all_methods()
    for metric in metrics:
        for expl in explanations: 
            keys.append(metric+"_"+expl)
    return keys

def get_krippen_keys():
    metrics = get_sub_metric_list(include_noncumulative=True)
    keys = []
    for metric in metrics:
            keys.append("kripp_"+metric)
    return keys

def get_test_size_red_keys():

    crit_dict_path = f"../results/critical_n_dict.npy"
    crit_dict = np.load(crit_dict_path,allow_pickle=True).item()

    def_model_id = list(crit_dict.keys())[0]
    def_group = list(crit_dict[def_model_id].keys())[0]
    def_config_path = list(crit_dict[def_model_id][def_group].keys())[0]

    crit_dict = crit_dict[list(crit_dict.keys())[0]]

    keys = []
    for group in crit_dict.keys():
        for metric in crit_dict[group][def_config_path]:
            keys.append(group+"_"+metric)

    return keys 

def test_training(config_path,model_id):

    exp_id = get_exp_id(config_path)

    path_train = glob.glob(f"../results/{exp_id}/metrics_{model_id}_train.csv")[0]
    path_val = glob.glob(f"../results/{exp_id}/metrics_{model_id}_val.csv")[0]
    path_test = glob.glob(f"../results/{exp_id}/metrics_{model_id}_test.csv")[0]

    train_acc = csv_to_perf(path_train,metric="accuracy")
    val_acc = csv_to_perf(path_val,metric="accuracy")
    test_acc = csv_to_perf(path_test,metric="accuracy")

    train_cal = csv_to_perf(path_train,metric="adaece_masked")
    val_cal = csv_to_perf(path_train,metric="adaece_masked")
    test_cal = csv_to_perf(path_test,metric="adaece_masked")

    return {"train_acc":train_acc,"val_acc":val_acc,"test_acc":test_acc,\
            "train_cal":train_cal,"val_cal":val_cal,"test_cal":test_cal}

def test_expl(config_path,model_id):

    result_dic = {}

    background_func = get_background_func(None)

    exp_id = get_exp_id(config_path)

    db_path = f"../results/{exp_id}/saliency_metrics.db"
    con = sqlite3.connect(db_path) 
    cur = con.cursor()

    metrics = get_sub_metric_list(True,True)
    explanations = get_all_methods()

    for metric in metrics:
        background = background_func(metric)

        for expl in explanations:

            query = f'SELECT metric_value FROM metrics WHERE model_id=="{model_id}" and metric_label=="{metric}" and replace_method=="{background}" and post_hoc_method=="{expl}"'
            perf_str = cur.execute(query).fetchone()[0]
            perf_mean = fmt_value_str(perf_str).mean()

            result_dic[metric+"_"+expl] = perf_mean

    return result_dic

def format_kripp_row(row):
    return [float(val.split(" ")[0]) for val in row.split(",")[1:]]

def test_kripp(config_path,model_id):

    exp_id = get_exp_id(config_path)

    with open( f"../results/{exp_id}/krippendorff_alpha_{model_id}_bNone_all.csv") as file:
        lines = [line.rstrip() for line in file]
        header,row1,row2 = lines[:-1]

    if (row1.split(",")[0]=="True") and row2.split(",")[0]=="False":
        cumulative,noncum = row1,row2 
    elif (row1.split(",")[0]=="False") and row2.split(",")[0]=="True":
        cumulative,noncum = row2,row1
    else:
        raise ValueError("Invalid formatting of the krippendorf's alpha csv file.")

    metrics_cumulative = header.split(",")[1:]
    metrics_non_cumulative = get_sub_cumulative_metric_list()
    cumulative = format_kripp_row(cumulative)
    noncum = format_kripp_row(noncum)

    result_dic = {}
    for metric_suff,value_list,metrics in zip(["","-nc"],[cumulative,noncum],[metrics_cumulative,metrics_non_cumulative]):
        for i in range(len(metrics)):
            result_dic["kripp_"+metrics[i]+metric_suff] = value_list[i]

    return result_dic
    

def test_test_size_red(config_path,model_id):

    crit_dict_path = f"../results/critical_n_dict.npy"
    crit_dict = np.load(crit_dict_path,allow_pickle=True).item()
    crit_dict = crit_dict[model_id]

    result_dic = {}

    for group in crit_dict.keys():
        for metric in crit_dict[group][config_path]:
            result_dic[group+"_"+metric] = crit_dict[group][config_path][metric]

    return result_dic

def get_step_dict(model_id,config_path,debug_arg):
    step_dict = {"training":{"args":get_training_args(model_id,config_path,debug_arg),"test_func":test_training},
                "expl":{"args":get_expl_args(model_id,config_path,debug_arg),"test_func":test_expl},
                "krippen":{"args":get_krippen_args(model_id,config_path),"test_func":test_kripp},
                "test_size_red":{"args":get_test_size_red_args(model_id,config_path,debug_arg),"test_func":test_test_size_red}}
    return step_dict

def get_training_args(model_id,config_path,debug_arg):
    training_args = ["python3","train_test.py","--model_ids",model_id,"--config_paths",config_path,"--val_batch_size","80"]
    if debug_arg != "":
        training_args.append(debug_arg)
    return training_args    

def get_expl_args(model_id,config_path,debug_arg):
    expl_arg = ["python3","compute_and_evaluate_explanations.py","--noise_tunnel_batchsize","1","--val_batch_size","80","--config_paths",config_path,"--model_ids",model_id]
    if debug_arg != "":
        expl_arg.append(debug_arg)
    return expl_arg

def get_krippen_args(model_id,config_path):
    krippen_arg = ["python3","krippendorf_alpha.py","--config_paths",config_path,"--model_ids",model_id,"--post_hoc_groups",'all',"--krippendorf_sample_nb","5","--model_id_baseline",model_id]    
    return krippen_arg

def get_test_size_red_args(model_id,config_path,debug_arg):
    test_size_reduction =["python3","test_size_reduction.py","--multi_processes_mode","--model_ids",model_id,"--config_paths",config_path,"--model_id_baseline",model_id]
    if debug_arg != "":
        test_size_reduction.append(debug_arg)
    return test_size_reduction 

def main(argv=None):

    argreader = ArgReader(argv)
    argreader.parser.add_argument('--config_path', type=str,default="configs/model_cub26debug.config")
    argreader.parser.add_argument('--steps_to_run', nargs="*",type=str,default=["training"])   

    argreader.getRemainingArgs()
    args = argreader.args

    if args.model_ids is None:
        args.model_ids = ["FL+FP+AP"]

    if args.debug:
        debug_arg = "--debug"
    else:
        debug_arg = ""

    commit = get_commit()

    keys_dic = get_keys_dic()

    for step in args.steps_to_run:   
        result_dic = {"commit":commit,"debug":args.debug}

        result_file_path,result_file_header = init_result_file(args.config_path,step,keys_dic)
        csv_rows = []
        
        for model_id in args.model_ids:
            print("MODEL ID",model_id)
            result_dic["model_id"] = model_id
            
            step_dict = get_step_dict(model_id,args.config_path,debug_arg)

            run_step(step_dict,step)
            result_dic.update(step_dict[step]["test_func"](args.config_path,model_id))

            csv_row = make_csv_row(result_file_header,result_dic)
            csv_rows.append(csv_row)

        with open(result_file_path,"a") as file:
            for row in csv_rows:
                print(row,file=file)

if __name__ == "__main__":
    main()

