import json
import os

from args import ArgReader,load_config,add_missing_args,set_batch_size
from training.trainer import train,addAllTrainingArgs
from utils import get_configs,get_model_ids

def load_config_and_model_args(config_path,model_id,model_args_dic):
    _args = load_config(config_path)

    _args.model_id = model_id

    for key in model_args_dic[model_id]:
        setattr(_args,key,model_args_dic[model_id][key])

    return _args

def main(argv=None):
    # Getting arguments from config file and command line
    # Building the arg reader
    argreader = ArgReader(argv)
    argreader = addAllTrainingArgs(argreader)
    argreader.getRemainingArgs()
    args = argreader.args

    config_paths = get_configs(args.config_paths)

    with open(args.model_args_path) as json_file:
        model_args_dic = json.load(json_file)

    model_ids = get_model_ids(args.model_ids,args.model_args_path,only_known_model_ids=True)

    for config_path in config_paths:
        print("CONFIG",config_path)

        for model_id in model_ids:
            _args = load_config_and_model_args(config_path,model_id,model_args_dic)
            _args = add_missing_args(_args,args)
            test_result_path = f"{_args.output_dir}/results/{_args.exp_id}/metrics_{_args.model_id}_test.csv"
            
            _args = set_batch_size(_args)

            if not os.path.exists(test_result_path):
            
                train(_args)
            else:
                print("Training of",_args.model_id,"already done.")

if __name__ == "__main__":
    main()
