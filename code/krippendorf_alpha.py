
import glob
import os 

from args import ArgReader,load_config,add_missing_args
from krippendorf.krippendorf import addArgs as kripp_args, compute_krippendorf_alpha
from krippendorf.krippendorf_aggregation import krippendorf_aggregation,addArgs as kripp_var_args
from krippendorf.krippendorf_aggregation_latex_table import make_latex_table_and_figure
from post_hoc_expl.utils import get_all_group_names
from utils import get_model_ids,get_configs
def main(argv=None):

    argreader = ArgReader(argv)
    argreader = kripp_args(argreader)
    argreader = kripp_var_args(argreader)
    
    argreader.parser.add_argument('--post_hoc_groups', type=str,nargs="*",help="If this arg is not set, all groups are used by default.")
    argreader.getRemainingArgs()

    args = argreader.args

    assert args.model_id_baseline is not None,"Set the id of the baseline model"

    config_paths = get_configs(args.config_paths,args.config_fold)

    model_ids = get_model_ids(args.model_ids,args.model_args_path)

    if args.post_hoc_groups is None:
        post_hoc_groups = get_all_group_names(include_recent=True,include_all=True)
    else:
        post_hoc_groups = args.post_hoc_groups

    print("Computing krippendorf's alpha")
    for config_path in config_paths:
        _args = load_config(config_path)
        _args = add_missing_args(_args,args)
        
        for model_id in model_ids:

            for post_hoc_group in post_hoc_groups:
                
                compute_krippendorf_alpha(model_id,post_hoc_group,args.krippendorf_sample_nb,_args,ordinal_metric=True,background=None,output_dir=args.output_dir)

    print("Computing krippendorf's aggregation")

    files_to_remove = glob.glob(f"{args.output_dir}/results/krippendorff_*_{args.model_id_baseline}_*.csv")
    for path in files_to_remove:
        os.remove(path)

    for model_id in model_ids: 
        for post_hoc_group in post_hoc_groups:
            krippendorf_aggregation(model_id,args.model_id_baseline,post_hoc_group,config_paths,args)
            
    print("Writing latex table")
    

    make_latex_table_and_figure(post_hoc_groups,args)
    
if __name__ == "__main__":
    main()