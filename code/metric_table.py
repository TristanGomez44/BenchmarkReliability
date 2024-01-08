from args import ArgReader
import configparser 

import numpy as np

from utils import get_configs,get_model_ids,get_dataset_labels

def model_metric_order():
    metrics = ["accuracy","accuracy_masked","sparsity","loss_ce","loss","ece","adaece","classece","ece_masked","adaece_masked","classece_masked","temperature"]
    return {metrics[i]:i for i in range(len(metrics))}

def get_model_metrics(model_metrics_csv_path):
    csv = np.genfromtxt(model_metrics_csv_path, delimiter=",",dtype=str)
    metrics = csv[0,1:]
    values = csv[-1,1:].astype(float)
    model_metric_order_dic = model_metric_order()
    metric_and_values = zip(metrics,values)
    metric_and_values = filter(lambda x:x[0] in model_metric_order_dic,metric_and_values)
    metrics,values = zip(*sorted(metric_and_values,key=lambda x:model_metric_order_dic[x[0]]))
    return {metrics[i]:values[i] for i in range(len(metrics))}

def get_best_perf_dic(model_ids,config_paths,metric_name,is_best,output_dir):
    best_perf_dic = {}
    for model_id in model_ids:
        for config_path in config_paths: 
            csv_path = get_csv_path(model_id,config_path,output_dir)
            model_metrics_dic = get_model_metrics(csv_path)
            metric_value = model_metrics_dic[metric_name]

            if config_path not in best_perf_dic.keys():
                best_perf_dic[config_path] = {"model_id":[model_id],metric_name:metric_value}

            elif is_best(metric_value,best_perf_dic[config_path][metric_name]):
                best_perf_dic[config_path] = {"model_id":[model_id],metric_name:metric_value}

            elif metric_value == best_perf_dic[config_path][metric_name]:
                best_perf_dic[config_path]["model_id"].append(model_id)
                
    return best_perf_dic

def get_csv_path(model_id,config_path,output_dir):
    config = configparser.ConfigParser()
    config.read(config_path)
    exp_id = config["default"]["exp_id"]
    csv_path = f"{output_dir}/results/{exp_id}/metrics_{model_id}_test.csv"
    return csv_path

def main(argv=None):

    argreader = ArgReader(argv)
    argreader.parser.add_argument('--metric', type=str,default="accuracy")

    argreader.getRemainingArgs()

    #Getting the args from command line and config file
    args = argreader.args

    model_ids = get_model_ids(args.model_ids,args.model_args_path)

    config_paths = get_configs(args.config_paths,args.config_fold)

    assert args.metric in ["accuracy","adaece"]

    if args.metric == "accuracy":
        is_best = np.greater
    elif "ece" in args.metric:
        is_best = np.less
    
    metric_list = [args.metric,args.metric+"_masked"]
    best_perf_dic = {}
    for metric in metric_list:
        best_perf_dic[metric] = get_best_perf_dic(model_ids,config_paths,metric,is_best,args.output_dir)

    prec_label= None

    dataset_labels = get_dataset_labels(config_paths)

    latex_table = "\\begin{tabular}{l" + "|cc"*len(config_paths) + "} \n"
    latex_table += "\\toprule \\\\ \n"
    latex_table += " & " + " & ".join(["\multicolumn{2}{c}{"+dataset_label+"}" for dataset_label in dataset_labels]) + "\\\\ \n"
    latex_table += " & Regular & Masked "*len(config_paths) + "\\\\ \n"
    latex_table += "\\midrule \\\\ \n"
    
    for model_id in model_ids:

        label = model_id

        if (prec_label is not None) and (len(label.split("+"))>len(prec_label.split("+"))):
            latex_table += "\\hline \n"
        prec_label = label

        latex_table += label + " & "

        for config_path in config_paths:

            csv_path = get_csv_path(model_id,config_path,args.output_dir)

            model_metrics_dic = get_model_metrics(csv_path)
            
            for metric in metric_list:
                fmt_str = str(round(100*model_metrics_dic[metric],1))
                if model_id in best_perf_dic[metric][config_path]["model_id"]:
                    latex_table += "\\textbf{" + fmt_str + "} & "
                else:
                    latex_table += fmt_str + " & "
            
        latex_table = latex_table[:-2] + "\\\\ \n"

    latex_table += "\\bottomrule \n"
    latex_table += "\\end{tabular} \n"

    with open(f"{args.output_dir}/results/{args.metric}_table.tex","w") as f:
        f.write(latex_table)

if __name__ == "__main__":
    main()