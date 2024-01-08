from args import ArgReader,load_config,addInitArgs,init_post_hoc_arg,addLossTermArgs,add_missing_args,set_batch_size

from post_hoc_expl.utils import get_all_methods
from utils import get_model_ids,get_configs
from data.load_data import addArgs as add_data_args
from model.modelBuilder import addArgs as add_model_args
from metrics.faithfulness_metrics import get_metric_types,get_is_multi_step_dic
from metrics.compute_scores_for_saliency_metrics import compute_scores
from metrics.compute_saliency_metrics import compute_sal_metrics

def main(argv=None):

    argreader = ArgReader(argv)
    argreader = addInitArgs(argreader)
    argreader = init_post_hoc_arg(argreader)
    argreader = add_model_args(argreader)
    argreader = add_data_args(argreader)
    argreader = addLossTermArgs(argreader)

    argreader.getRemainingArgs()

    #Getting the args from command line and config file
    args = argreader.args

    config_paths = get_configs(args.config_paths,args.config_fold)

    metric_types= get_metric_types(include_noncumulative=False)
    is_multi_step_dic = get_is_multi_step_dic()

    explanation_methods = get_all_methods()

    model_ids = get_model_ids(args.model_ids,args.model_args_path)

    for config_path in config_paths:
 
        _args = load_config(config_path)
        _args.noise_tunnel_batchsize = args.noise_tunnel_batchsize
        _args.val_batch_size = args.val_batch_size
        _args.debug  = args.debug
        _args = add_missing_args(_args,args)
        
        for metric_type in metric_types:

            for explanation_method in explanation_methods:

                for model_id in model_ids:

                    _args = set_batch_size(_args,model_id)

                    for cumulative in [True,False]:

                        if cumulative or is_multi_step_dic[metric_type]:
                            
                            compute_scores(metric_type,explanation_method,model_id,cumulative,_args)
        

        _args.model_ids = model_ids
        compute_sal_metrics(_args.exp_id,_args)

if __name__ == "__main__":
    main()