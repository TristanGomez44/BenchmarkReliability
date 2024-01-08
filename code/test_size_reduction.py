import os,sys
import configparser
from itertools import product
import collections
import math
import multiprocessing

import numpy as np
from scipy.stats import multinomial

from args import ArgReader
from data.load_data import get_class_nb

from krippendorf.krippendorf import fmt_metric_values,get_background_func,convert_to_ordinal,keep_required_explanations,preprocc_matrix
from krippendorf.krippendorf_aggregation import order_metrics
from krippendorf.krippendorf_aggregation_latex_table import get_post_hoc_dic
from metrics.faithfulness_metrics import get_sub_metric_list,get_metrics_to_minimize
from metrics.compute_saliency_metrics import get_db
import post_hoc_expl.utils as post_hoc_utils 
from utils import get_model_ids,get_configs,get_dataset_labels,Tree,plot_difference_matrix_as_figure

EPS = 1e-6

def normalize_critical_n_dict(critical_n_dict,max_n_dict,model_ids,post_hoc_groups,config_paths,metrics):
    for model_id in model_ids:
        for post_hoc_group in post_hoc_groups:
            for config_path in config_paths:
                config_filename = os.path.basename(config_path)
                for metric in metrics:
                    critical_n = critical_n_dict[model_id][post_hoc_group][config_filename][metric]
                    max_n = max_n_dict[config_filename]
                    
                    normalized_critical_n = (critical_n)/max_n
                    critical_n_dict[model_id][post_hoc_group][config_filename][metric] = normalized_critical_n
    
    return critical_n_dict 



def add_row(latex_txt,post_hoc_groups,aggr_dic,model_id,best_perf_dic):
    for i,group in enumerate(post_hoc_groups):
        perf = aggr_dic[model_id][group]
        perf_fmt = str(round(100*perf,1))
        if perf==best_perf_dic[group]:
            latex_txt += "\\textbf{"+perf_fmt+"}"
        else:
            latex_txt += perf_fmt 
        if i < len(post_hoc_groups)-1:
            latex_txt += "&"
    return latex_txt 

def plot_and_aggregate_difference_matrices(critical_n_dict,model_ids,model_id_baseline,post_hoc_groups,metrics,config_paths,aggr_func_name,file_pref="test_size_reduction",output_dir="../"):

    aggr_func = np.mean if aggr_func_name == "mean" else np.median

    aggr_dic = {}

    dataset_labels = get_dataset_labels(config_paths)

    config_filenames = [os.path.basename(path) for path in config_paths]

    for post_hoc_group in post_hoc_groups:
        
        baseline_matrix = make_matrix_to_plot(critical_n_dict[model_id_baseline][post_hoc_group],config_filenames,metrics)

        for model_id in model_ids:
            
            if not model_id in aggr_dic:
                aggr_dic[model_id] = {}
        
            matrix = make_matrix_to_plot(critical_n_dict[model_id][post_hoc_group],config_filenames,metrics)
            aggr_dic[model_id][post_hoc_group] = aggr_func(matrix)

            if model_id !=  model_id_baseline:

                metrics,(matrix,baseline_matrix) = order_metrics(metrics,matrix,baseline_matrix)

                metric_labels = [metric.upper() for metric in metrics]
                filename = file_pref+"_"+model_id+"_"+model_id_baseline+"_"+post_hoc_group+".png"
                plot_difference_matrix_as_figure((100*matrix).astype(int),(100*baseline_matrix).astype(int),metric_labels,dataset_labels,filename,output_dir=output_dir,min_diff=-100,max_diff=100,inv_diff=True,color_bar=False,aspect_ratio=0.75,fontsize=20,scale=10)

    post_hoc_dic = get_post_hoc_dic()
    post_hoc_labels = [post_hoc_dic[post_hoc_group] for post_hoc_group in post_hoc_groups]
    
    latex_txt = "\\begin{tabular}{c|"+len(post_hoc_labels)*"c"+"}"+"\\\\ \n"
    latex_txt += "\\toprule \n"
    latex_txt += "&"+ "&".join(post_hoc_labels)+"\\\\ \n"
    latex_txt += "\\midrule \n"
    prev_label = None

    best_perf_dic = {}
    for group in post_hoc_groups:
        best_perf_dic[group] = min([aggr_dic[model_id][group] for model_id in model_ids])

    for model_id in model_ids:
        model_label = model_id

        latex_txt += model_label+"&"
        latex_txt = add_row(latex_txt,post_hoc_groups,aggr_dic,model_id,best_perf_dic)
        latex_txt += "\\\\ \n"

        if model_label == "Baseline":
            latex_txt += "\\hline \n"
        elif prev_label is not None and len(model_label.split("+"))>len(prev_label.split("+")):
            latex_txt += "\\hline \n"

        prev_label = model_label

    latex_txt += "\\bottomrule \n"
    latex_txt += "\\end{tabular} \n"

    print(os.path.join(output_dir,"results",file_pref+".tex"))
    with open(os.path.join(output_dir,"results",file_pref+".tex"),"w") as file:
        print(latex_txt,file=file)
    
def make_matrix_to_plot(sub_dict,key_list1,key_list2):
    matrix = np.zeros((len(key_list1),len(key_list2)))
    for j,key1 in enumerate(key_list2):
        for i,key2 in enumerate(key_list1):
            if isinstance(sub_dict[key1][key2],np.ndarray) and len(sub_dict[key1][key2].shape)>0:
                matrix[i,j] = sub_dict[key2][key1].mean()
            else:
                matrix[i,j] = sub_dict[key2][key1]

    return matrix

def recursively_default_dict():
    return collections.defaultdict(recursively_default_dict)

def rank_probs(matrix,rank_to_count=1):
    counts = (matrix==rank_to_count).sum(axis=0)
    probs = counts/counts.sum()
    return probs

def store_results_in_critical_n_dict(critical_n_list,args_list,critical_n_dict):

    for critical_n,args in zip(critical_n_list,args_list):
        _,_,_,_,_,model_id,post_hoc_group,config_path,metric = args 
        critical_n_dict[model_id][post_hoc_group][config_path][metric] = critical_n 

    return critical_n_dict

def critical_N_worker(probs,max_n,prob_thresh,frac_to_sample,critical_n_dict,model_id,post_hoc_group,config_path,metric):
    print("Start critical N search for",model_id,post_hoc_group,config_path,metric)
    critical_N = search_critical_N(probs,max_n,prob_thresh,frac_to_sample)
    return critical_N

def search_critical_N(probs,max_n,prob_thresh=0.95,frac_to_sample=0.1):

    probs = np.array(probs)

    assert abs(probs.sum()-1)<EPS,"Argument probs should approximately sum to 1"

    max_prob = probs.max()

    if (probs==max_prob).sum() > 1:
        print("There is several best performing method. Impossible to reduce test size.")
        return max_n
    else:
       max_ind = probs.argmax()

    thres_found = False 
    start = 0
    end = max_n

    ratio = 1/frac_to_sample
    if ratio != round(ratio):
        print("Warning: ratio is not an integer:",ratio)

    last_valid_n = None

    #Searching critical current_n by dichotomy

    while (not thres_found) and abs(start-end)>1:
        
        current_n = math.ceil((start+end)/2)

        rv = multinomial(current_n, probs)

        prob_list = []
        is_valid_combination = []

        for counts in product(range(0,current_n+1), repeat=len(probs)):
            if sum(counts) == current_n:
                if counts[0]%ratio==0:

                    prob = rv.pmf(counts)
                    prob_list.append(prob)

                    is_valid = True
                    for i in range(len(probs)):
                        if i != max_ind:
                            is_valid = is_valid & (counts[max_ind]>counts[i])

                    is_valid_combination.append(is_valid)

        prob_list = np.array(prob_list)
        is_valid_combination = np.array(is_valid_combination)
        prob = prob_list[is_valid_combination].sum()/prob_list.sum()

        if prob>=prob_thresh:
            last_valid_n = current_n
            end = current_n
        else:
            start = current_n

    if last_valid_n is None:
        print("Could not reduce test size. Returning n_max")
        best_n = max_n 
    else:
        best_n = last_valid_n

    return best_n

def main(argv=None):

    argreader = ArgReader(argv)
    argreader.parser.add_argument('--background', type=str)
    argreader.parser.add_argument('--model_id_baseline', type=str,default="Baseline")  
    argreader.parser.add_argument('--prob_thres', type=float,default=0.95)  
    argreader.parser.add_argument('--frac_to_sample_per_method', type=int,default=0.5) 
    argreader.parser.add_argument('--multi_processes_mode', action="store_true") 
    argreader.parser.add_argument('--aggr_func_name',type=str,default="mean")     
    argreader.getRemainingArgs()
    args = argreader.args

    assert args.aggr_func_name in ["mean","median"],"Invalid aggregation function. Choose mean or median."

    config_paths = get_configs(args.config_paths,args.config_fold)

    model_ids = get_model_ids(args.model_ids,args.model_args_path)
 
    post_hoc_groups = ["recent"]+post_hoc_utils.get_all_group_names()

    metrics = get_sub_metric_list()
    background_func = get_background_func(args.background)

    metrics_to_minimize = get_metrics_to_minimize()

    critical_n_dict = Tree()
    critical_n_dict_path = os.path.join(args.output_dir,"results","critical_n_dict.npy")

    if os.path.exists(critical_n_dict_path):
        critical_n_dict = np.load(critical_n_dict_path,allow_pickle=True).item()

    if args.multi_processes_mode:
        args_list = []

    for model_id in model_ids:
        
        for post_hoc_group in post_hoc_groups:
            dataset_test_list = []
            for config_path in config_paths:

                config = configparser.ConfigParser()
                config.read(config_path)
                exp_id = config["default"]["exp_id"]
                dataset_test = config["default"]["dataset_test"]
                dataset_test_list.append(dataset_test)
                _,curr = get_db(exp_id,output_dir=args.output_dir)

                config_filename = os.path.basename(config_path)

                for metric in metrics:

                    if (args.debug) or not metric in critical_n_dict[model_id][post_hoc_group][config_filename]:

                        background = background_func(metric)

                        query = f'SELECT post_hoc_method,metric_value FROM metrics WHERE model_id=="{model_id}" and metric_label=="{metric}" and replace_method=="{background}" and post_hoc_method != ""'

                        output = curr.execute(query).fetchall()

                        output = keep_required_explanations(output,post_hoc_group,post_hoc_col_ind=0)
                        _,metric_values_list = zip(*output)

                        metric_values_matrix = fmt_metric_values(metric_values_list)
                        metric_values_matrix = preprocc_matrix(metric_values_matrix,metric)

                        metric_values_matrix = convert_to_ordinal(metric_values_matrix,metric,metrics_to_minimize)

                        probs = rank_probs(metric_values_matrix)
                        max_n = len(metric_values_matrix)

                        if args.multi_processes_mode:
                            worker_args = (probs,max_n,args.prob_thres,args.frac_to_sample_per_method,critical_n_dict,model_id,post_hoc_group,config_filename,metric)    
                            args_list.append(worker_args)
                        else:
                            critical_n = search_critical_N(probs,max_n,args.prob_thres,args.frac_to_sample_per_method)
                            critical_n_dict[model_id][post_hoc_group][config_filename][metric] = critical_n
            
        if args.multi_processes_mode:
            pool = multiprocessing.Pool()
            critical_n_list = pool.starmap(critical_N_worker,args_list)
            critical_n_dict = store_results_in_critical_n_dict(critical_n_list,args_list,critical_n_dict)

        np.save(critical_n_dict_path,critical_n_dict)

    dataset_test_list = []
    max_n_dict = {}
    for config_path in config_paths:
        config = configparser.ConfigParser()
        config.read(config_path)
        config = config["default"]
        dataset_test = config["dataset_test"]
        dataset_test_list.append(dataset_test)
        dataset_train = config["dataset_train"]
        max_n = int(config["img_nb_per_class"])*get_class_nb(args.data_dir,dataset_train)
        max_n_dict[os.path.basename(config_path)] = max_n

    critical_n_dict = normalize_critical_n_dict(critical_n_dict,max_n_dict,model_ids,post_hoc_groups,config_paths,metrics)

    plot_and_aggregate_difference_matrices(critical_n_dict,model_ids,args.model_id_baseline,post_hoc_groups,metrics,config_paths,args.aggr_func_name,output_dir=args.output_dir)

if __name__ == "__main__":
    main()