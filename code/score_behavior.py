import configparser

import numpy as np

from args import ArgReader
from krippendorf.krippendorf import get_background_func,keep_required_explanations
from metrics.faithfulness_metrics import get_metric_types
from metrics.compute_saliency_metrics import get_db
from utils import get_configs,get_model_ids,plot_difference_matrix_as_figure
def main(argv=None):

    argreader = ArgReader(argv)
    argreader.parser.add_argument('--background', type=str)
    argreader.parser.add_argument('--model_id_baseline', type=str,default="Baseline")  
    argreader.getRemainingArgs()
    args = argreader.args

    config_paths = get_configs(args.config_paths,args.config_fold)

    model_ids = get_model_ids(args.model_ids,args.model_args_path)
 
    metrics = get_metric_types(include_noncumulative=False)
    background_func = get_background_func(args.background)

    mat_dict = {}

    for model_id in model_ids:
        dataset_test_list = []

        matrix = []

        for config_path in config_paths:

            config = configparser.ConfigParser()
            config.read(config_path)
            exp_id = config["default"]["exp_id"]

            dataset_test = config["default"]["dataset_test"]
            dataset_test_list.append(dataset_test)
            _,curr = get_db(exp_id,output_dir=args.output_dir)

            row = []

            for metric in metrics:

                background = background_func(metric)

                query = f'SELECT post_hoc_method,metric_value FROM metrics WHERE model_id=="{model_id}" and metric_label=="{metric.upper()}_VAL_RATE" and replace_method=="{background}" and post_hoc_method != ""'

                output = curr.execute(query).fetchall()
                
                output = keep_required_explanations(output,"all",post_hoc_col_ind=0)
                _,metric_values_list = zip(*output)

                mean = np.array(metric_values_list).astype(float).mean()

                row.append(mean)

            matrix.append(row)
        
        mat_dict[model_id] = np.array(matrix)
    
    baseline_mat = mat_dict[args.model_id_baseline]

    for model_id in model_ids:

        if model_id != args.model_id_baseline:
        
            mat = mat_dict[model_id]

            plot_difference_matrix_as_figure(100*mat,100*baseline_mat,metrics,dataset_test_list,f"score_behavior_{model_id}_{args.model_id_baseline}",output_dir=args.output_dir,min_diff=-25,max_diff=25)

if __name__ == "__main__":
    main()