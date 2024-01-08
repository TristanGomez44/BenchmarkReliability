import glob
import math 
from functools import partial

import numpy as np 
import matplotlib.pyplot as plt
import matplotlib
from metrics.faithfulness_metrics import get_sub_metric_list,get_sub_noncumulative_metric_list
from krippendorf.utils import get_aggr_kripp_dic

DATASET_DIC = {"CUB_200_2011":"CUB","CARS25":"Standford cars","CARS26":"Standford cars","cars":"Standford cars","aircraft":"FGVC","embryo_img":"Embryo","crohn_2":"Crohn"}

def get_metric(csv,key):
    col_index = csv[0].tolist().index(key)
    value = csv[-1,col_index].astype(float)
    return value

def get_calibration_and_accuracy(path):
    csv = np.genfromtxt(path, delimiter=",",dtype=str)
    calibration = get_metric(csv,"adaece_masked")
    accuracy =  get_metric(csv,"accuracy")
    accuracy_masked = get_metric(csv,"accuracy_masked")
    return accuracy,accuracy_masked,calibration

def get_csv(exp_id,model_id,background,post_hoc_group,output_dir):
    filename_suff = f"{model_id}_b{background}"
    csv_path = f"{output_dir}/results/{exp_id}/krippendorff_alpha_values_list_{filename_suff}_{post_hoc_group}.csv"
    csv = np.genfromtxt(csv_path, delimiter=",",dtype=str)
    return csv

def get_alpha(model_id,prefix):
    return float(model_id.split(prefix)[1])

def get_alphas(model_ids,prefix):
    alpha_list = [0] + [get_alpha(model_id,prefix) for model_id in model_ids[1:-1]] + [10]
    return alpha_list

def make_krippendorf_interpolation_figure(args):

    best_model_paths = sorted(glob.glob(f"{args.output_dir}/models/{args.exp_id}/model{args.model_id_prefix}*_best_epoch*"))
    model_ids = [path.split("/")[-1].split("_best")[0].replace("model","") for path in best_model_paths]

    get_interp_func = partial(get_alpha,prefix=args.model_id_prefix)

    model_ids = filter(lambda x:get_interp_func(x) not in [0,10],model_ids)
    model_ids = sorted(model_ids,key=partial(get_alpha,prefix=args.model_id_prefix))
    model_ids = [args.model_id_start] + list(model_ids) + [args.model_id_end]

    res_dict = None

    for model_id in model_ids:

        kripp_inter_dic = get_aggr_kripp_dic(args.exp_id,model_id,args.background,args.post_hoc_group,args.metrics_to_remove,args.output_dir)

        metrics = get_sub_metric_list()

        if res_dict is None:
            res_dict = {metric:{"calibration":[],"accuracy":[],"krippendorf":[],"krippendorf_low":[],"krippendorf_high":[]} for metric in metrics}

        for i in range(len(metrics)):

            metric = list(metrics)[i]
            mean_kripp = kripp_inter_dic[metric]["mean"]
            low_kripp,high_kripp = kripp_inter_dic[metric]["lower"],kripp_inter_dic[metric]["upper"]

            csv_path = f"{args.output_dir}/results/{args.exp_id}/metrics_{model_id}_test.csv"
            accuracy,_,calibration = get_calibration_and_accuracy(csv_path)

            res_dict[metric]["krippendorf"].append(mean_kripp)
            res_dict[metric]["krippendorf_low"].append(low_kripp)
            res_dict[metric]["krippendorf_high"].append(high_kripp)
            res_dict[metric]["calibration"].append(calibration)
            res_dict[metric]["accuracy"].append(accuracy)

    fig,axs = plt.subplots(len(metrics),2,figsize=(15,25))

    interp_alphas = get_alphas(model_ids,args.model_id_prefix)
    interp_alphas = np.array(interp_alphas)
    progress = (interp_alphas - np.min(interp_alphas))/(np.max(interp_alphas) - np.min(interp_alphas))
    cmap = matplotlib.colormaps['plasma']

    #One plot for each metric 
    for i,metric in enumerate(metrics):
        axs[i,0].scatter(res_dict[metric]["krippendorf"],res_dict[metric]["accuracy"],c=progress,cmap=cmap)
        axs[i,0].set_xlabel("Krippendorf's alpha")
        axs[i,0].set_ylabel("Accuracy")
        axs[i,0].set_title(metric.upper())

        axs[i,1].scatter(res_dict[metric]["krippendorf"],res_dict[metric]["calibration"],c=progress,cmap=cmap)
        axs[i,1].set_xlabel("Krippendorf's alpha")
        axs[i,1].set_ylabel("Calibration")
        axs[i,1].set_title(metric)

        #Add colorbar
        norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=axs[i,0],pad=0.1)
        
    plt.tight_layout()
    plt.savefig(f"{args.output_dir}/vis/{args.exp_id}/krippendorf_vs_focalloss_{args.model_id_prefix}.png")

    #Plot krippendorf's alpha as a function of the interpolation parameter for each metric 
    nrows = int(math.sqrt(len(metrics)))
    ncols = int(math.ceil(len(metrics)/nrows))
    fig,axs = plt.subplots(nrows,ncols,figsize=(10,5))

    for i,metric in enumerate(metrics):
        #Plot lines and points
        row = i//ncols
        col = i%ncols

        mean = np.array(res_dict[metric]["krippendorf"])
        low = np.array(res_dict[metric]["krippendorf_low"])
        high = np.array(res_dict[metric]["krippendorf_high"])

        axs[row,col].plot(interp_alphas,100*mean,marker="o")

        #set xticks
        axs[row,col].set_xticks([0,5,10])
        axs[row,col].set_xticklabels(["0","0.5","1"])
        
        axs[row,col].fill_between(interp_alphas,100*low,100*high,alpha=0.2)

        if row == nrows-1:
            axs[row,col].set_xlabel("Interpolation coeff. (β)")
        if col == 0:
            axs[row,col].set_ylabel("Krippendorf's alpha (x100)")
        axs[row,col].set_title(f"{metric} - {DATASET_DIC[args.exp_id]}")

    plt.tight_layout()
    plt.savefig(f"{args.output_dir}/vis/{args.exp_id}/krippendorf_vs_interp_{args.model_id_prefix}.png")