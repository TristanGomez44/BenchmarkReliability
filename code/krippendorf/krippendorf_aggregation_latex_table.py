import glob, os, json
import numpy as np 

from krippendorf.utils import order_groups
from utils import get_post_hoc_group_label_dict,plot_difference_matrix_as_figure

def get_post_hoc_dic():
    return {"all":"ALL","classmap":"Class-Map","backprop":"BP","perturb":"Perturbations","recent":"Recent"}

def load_csv(path):
    csv = np.genfromtxt(path, delimiter=",",dtype=str)
    csv = [csv] if len(csv.shape)==1 else csv
    return csv 

def make_latex_table_and_figure(groups,args,aggr_func="mean"):

    csv_file_paths = glob.glob(f"{args.output_dir}/results/krippendorff_{aggr_func}_diff_{args.model_id_baseline}_*.csv")

    print(csv_file_paths)

    aggr_dic = {}

    #Dic to store the model_id of the best model for each group
    best_model_id_dic = {}

    mean_matrix = []
    baseline_matrix = []
    groups = []

    for path in csv_file_paths:

        mean_row = []
        baseline_row = []
        model_ids = []

        group = os.path.splitext(os.path.basename(path))[0].split(args.model_id_baseline+"_")[1]
    
        groups.append(group)

        csv = load_csv(path)

        csv_mean = load_csv(path.replace("_diff",""))
        csv_baseline = load_csv(path.replace("diff","baseline"))

        for row,row_mean,row_baseline in zip(csv,csv_mean,csv_baseline):
            model_id,diff = row

            _,mean = row_mean
            _,baseline = row_baseline

            if model_id != args.model_id_baseline:
                mean_row.append(float(mean))
                baseline_row.append(float(baseline))
                model_ids.append(model_id)

            if model_id not in aggr_dic:
                aggr_dic[model_id] = {}
            
            aggr_dic[model_id][group] = float(diff)

            if group not in best_model_id_dic:
                best_model_id_dic[group] = model_id
            
            if aggr_dic[model_id][group] > aggr_dic[best_model_id_dic[group]][group]:
                best_model_id_dic[group] = model_id
    
        mean_matrix.append(np.array(mean_row))
        baseline_matrix.append(np.array(baseline_row))

    make_latex_table(groups,args,aggr_dic,best_model_id_dic,aggr_func)

    filename = f"krippendorff_{aggr_func}_diff_{args.model_id_baseline}.png"

    mean_matrix = np.stack(mean_matrix).transpose()
    baseline_matrix = np.stack(baseline_matrix).transpose()

    groups,(mean_matrix,baseline_matrix) = order_groups(groups,mean_matrix,baseline_matrix)

    post_hoc_group_label_dict = get_post_hoc_group_label_dict()

    group_labels = [post_hoc_group_label_dict[group] for group in groups]

    plot_difference_matrix_as_figure(mean_matrix,baseline_matrix,group_labels,model_ids,filename,output_dir=args.output_dir,min_diff=-10,max_diff=10,aspect_ratio=0.5,fontsize=20,scale=12,make_max_bold=True)

def make_latex_table(groups,args,aggr_dic,best_model_id_dic,aggr_func):

    with open(args.model_args_path) as json_file:
        model_ids = json.load(json_file).keys()

    group_dic = get_post_hoc_dic()
    
    csv = "\\begin{tabular}{c|"+len(groups)*"c"+"}"+"\\\\ \n"
    csv += "\\toprule \n"
    csv += "Model & "
    for i,group in enumerate(groups):
        csv += group_dic[group]
        
        if i < len(groups)-1:
            csv += " & "

    csv += "\\\\\n"

    csv += "\\midrule \n"

    prec_label = None

    for model_id in model_ids:
        
        if model_id in aggr_dic:
            
            label = model_id

            if (prec_label is not None) and (len(label.split("+"))>len(prec_label.split("+"))):
                csv += "\\hline \n"
            prec_label = label

            csv += label+" & "
            for i,group in enumerate(groups):
                
                #Writes in bold the best model for each group
                if model_id == best_model_id_dic[group]:
                    csv += "\\textbf{"
                
                csv += f"{aggr_dic[model_id][group]:.3f}"

                if model_id == best_model_id_dic[group]:
                    csv += "}"

                if i < len(groups)-1:
                    csv += " & "

            csv += "\\\\\n"

    csv += "\\bottomrule \n"

    csv += "\\end{tabular} \n"

    #saves csv in file
    with open(f"{args.output_dir}/results/krippendorff_{aggr_func}_diff_{args.model_id_baseline}.tex", "w") as text_file:
        text_file.write(csv)
