import glob, os
import configparser

import numpy as np 
import scipy.stats 
from krippendorf.utils import get_aggr_kripp_dic,order_metrics
from utils import get_dataset_label_dic,plot_difference_matrix_as_figure

def get_csv(exp_id,model_id,background,post_hoc_group,output_dir):
    filename_suff = f"{model_id}_b{background}"

    csv_path = f"{output_dir}/results/{exp_id}/krippendorff_alpha_values_list_{filename_suff}_{post_hoc_group}.csv"       

    if os.path.exists(csv_path):
        csv = np.genfromtxt(csv_path, delimiter=",",dtype=str)
    else:
        csv = None
    return csv

def format_values_for_latex(difference,font,color,arrow):
    difference = f"{difference:.1f}"
    return "\\textcolor{"+color+"}{$"+font+"{"+difference+"}"+arrow+"$}"

def krippendorf_aggregation(model_id,model_id_baseline,post_hoc_group,config_paths,args,background=None):

    dataset_dic = get_dataset_label_dic()

    baseline_mean_matrix = []
    mean_matrix = []
    difference_matrix = []
    is_significant_matrix = []
    dataset_test_list = []
    actual_config_paths =[]
    for config_path in config_paths:
        config = configparser.ConfigParser()
        config.read(config_path)

        exp_id = config["default"]["exp_id"]
        dataset_test = config["default"]["dataset_test"]
        
        csv_baseline = get_csv(exp_id,model_id_baseline,background,post_hoc_group,args.output_dir)
        csv = get_csv(exp_id,model_id,background,post_hoc_group,args.output_dir)

        aggr_kripp_dic_baseline = get_aggr_kripp_dic(exp_id,model_id_baseline,background,post_hoc_group,args.metrics_to_remove,args.output_dir)

        aggr_kripp_dic = get_aggr_kripp_dic(exp_id,model_id,background,post_hoc_group,args.metrics_to_remove,args.output_dir)

        if (csv_baseline is not None) and (csv is not None):

            row = dataset_dic[dataset_test.replace("_test","")]+"&"
            difference_row = []
            is_significant_row = []
            baseline_mean_row = []
            mean_row = []
            metrics=[]
            for i in range(len(csv)):

                metric = csv[i,0]
             
                if metric not in args.metrics_to_remove:
                    metrics.append(metric)
                    values_baseline = csv_baseline[i,1:].astype(float)
                    values = csv[i,1:].astype(float)
                    
                    if scipy.stats.shapiro(values)[1] < args.test_thres or scipy.stats.shapiro(values_baseline)[1] < args.test_thres:
                        is_significant = scipy.stats.mannwhitneyu(values,values_baseline)[1] < args.test_thres

                    else:
                        equal_var = scipy.stats.levene(values,values_baseline)[1] > args.test_thres
                        is_significant = scipy.stats.ttest_ind(values_baseline,values,equal_var=equal_var)[1] < args.test_thres

                    baseline_mean = aggr_kripp_dic_baseline[metric]["mean"]
                    mean = aggr_kripp_dic[metric]["mean"]

                    difference = 100*(mean - baseline_mean)
                    difference_row.append(difference)
                    is_significant_row.append(is_significant)

                    baseline_mean_row.append(100*baseline_mean)
                    mean_row.append(100*mean)

                    font = "\\mathbf" if is_significant else ""
                    color = "green" if difference > 0 else "red"
                    arrow = "\\uparrow" if difference > 0 else "\\downarrow"

                    string = format_values_for_latex(difference,font,color,arrow)

                    row += string
                    
                    if i < len(csv_baseline) - 1:
                        row += "&"

            difference_matrix.append(difference_row)
            is_significant_matrix.append(is_significant_row)
            baseline_mean_matrix.append(baseline_mean_row)
            mean_matrix.append(mean_row)

            actual_config_paths.append(os.path.basename(config_path))
            dataset_test_list.append(dataset_test)

            row += "\\\\"

            out_path = args.output_dir+"/results/"+args.out_file_name

            #Check if file exists. If not, write header 
            if not glob.glob(out_path):
                with open(out_path,"w") as f:
                    print("\\begin{tabular}{c|"+len(metrics)*"c"+"}"+"\\\\",file=f)
                    print("Dataset&"+"&".join(metrics)+"\\\\",file=f)
                    print("\\hline",file=f)

            with open(out_path,"a") as f:
                print(row,file=f)

    metrics,(mean_matrix,baseline_mean_matrix,difference_matrix,is_significant_matrix) = order_metrics(metrics,mean_matrix,baseline_mean_matrix,difference_matrix,is_significant_matrix)

    metric_labels = [metric.upper() for metric in metrics]
    dataset_labels = [dataset_dic[dataset_test.replace("_test","")] for dataset_test in dataset_test_list]
    filename = "krippendorff_heatmap_"+model_id+"_"+model_id_baseline+"_"+post_hoc_group+".png"
    plot_difference_matrix_as_figure(mean_matrix,baseline_mean_matrix,metric_labels,dataset_labels,filename,is_significant_matrix,args.output_dir,aspect_ratio=0.5,fontsize=20,scale=15,min_diff=-45,max_diff=45,color_bar=False)

    for aggr_func,aggr_func_label in zip([np.mean,np.median],["mean","median"]):

        for matrix,mat_label in zip([mean_matrix,baseline_mean_matrix,difference_matrix],["","_baseline","_diff"]):

            aggr_matrix = aggr_func(matrix)

            csv_path = f"{args.output_dir}/results/krippendorff_{aggr_func_label}{mat_label}_{model_id_baseline}_{post_hoc_group}.csv"
            with open(csv_path,"a") as f:
                print(f"{model_id},{aggr_matrix}",file=f)

def addArgs(argreader):
    argreader.parser.add_argument('--model_id_baseline', type=str,default="Baseline")
    argreader.parser.add_argument('--test_thres', type=float,default=0.05)
    argreader.parser.add_argument('--out_file_name', type=str,default="krippendorff.tex")
    argreader.parser.add_argument('--metrics_to_remove', type=str,nargs="*",default=["DAUC-nc","IAUC-nc","IIC"])
    return argreader
