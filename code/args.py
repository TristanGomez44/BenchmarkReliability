import sys,os
import argparse
import configparser
from argparse import Namespace

import torch

def add_missing_args(_args,args):

    args_dic = vars(args)
    _args_dic = vars(_args)

    for key in args_dic:
        if key not in _args_dic:
            _args_dic[key] = args_dic[key]

    return Namespace(**_args_dic)

def str2bool(v):
    '''Convert a string to a boolean value'''
    if v == 'True':
        return True
    elif v == 'False':
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
def str2FloatList(x):

    '''Convert a formated string to a list of float value'''
    if len(x.split(",")) == 1:
        return float(x)
    else:
        return [float(elem) for elem in x.split(",")]
def strToStrList(x):
    if x == "None":
        return []
    else:
        return x.split(",")

def str2StrList(x):
    '''Convert a string to a list of string value'''
    return x.split(" ")

class ArgReader():
    """
    This class build a namespace by reading arguments in both a config file
    and the command line.

    If an argument exists in both, the value in the command line overwrites
    the value in the config file

    This class mainly comes from :
    https://stackoverflow.com/questions/3609852/which-is-the-best-way-to-allow-configuration-options-be-overridden-at-the-comman
    Consulted the 18/11/2018

    """

    def __init__(self,argv):
        ''' Defines the arguments used in several scripts of this project.
        It reads them from a config file
        and also add the arguments found in command line.

        If an argument exists in both, the value in the command line overwrites
        the value in the config file
        '''

        # Do argv default this way, as doing it in the functional
        # declaration sets it at compile time.
        if argv is None:
            argv = sys.argv

        # Parse any conf_file specification
        # We make this parser with add_help=False so that
        # it doesn't parse -h and print help.
        conf_parser = argparse.ArgumentParser(
            description=__doc__, # printed with -h/--help
            # Don't mess with format of description
            formatter_class=argparse.RawDescriptionHelpFormatter,
            # Turn off help, so we print all options in response to -h
            add_help=False
            )
        conf_parser.add_argument("-c", "--conf_file",
                            help="Specify config file", metavar="FILE")
        args, self.remaining_argv = conf_parser.parse_known_args()

        defaults = {}

        if args.conf_file:
            config = configparser.ConfigParser()
            config.read([args.conf_file])
            defaults.update(dict(config.items("default")))

        # Parse rest of arguments
        # Don't suppress add_help here so it will handle -h
        self.parser = argparse.ArgumentParser(
            # Inherit options from config_parser
            parents=[conf_parser]
            )
        self.parser.set_defaults(**defaults)

        self.parser.add_argument('--log_interval', type=int, metavar='M',
                            help='The number of batchs to wait between each console log')
        self.parser.add_argument('--num_workers', type=int,metavar='NUMWORKERS',
                            help='the number of processes to load the data. num_workers equal 0 means that it’s \
                            the main process that will do the data loading when needed, num_workers equal 1 is\
                            the same as any n, but you’ll only have a single worker, so it might be slow')
        self.parser.add_argument('--cuda', type=str2bool, metavar='S',
                            help='To run computations on the gpu')
        self.parser.add_argument('--multi_gpu', type=str2bool, metavar='S',
                            help='If cuda is true, run the computation with multiple gpu')
        self.parser.add_argument('--debug', action="store_true",
                            help="To run only a few batch of training and a few batch of validation")

        self.parser.add_argument('--epochs', type=int, metavar='N',
                            help='maximum number of epochs to train')

        #Arg to share
        self.parser.add_argument('--model_id', type=str, metavar='IND_ID',
                            help='the id of the individual model')
        self.parser.add_argument('--exp_id', type=str, metavar='EXP_ID',
                            help='the id of the experience')
        self.parser.add_argument('--seed', type=int, metavar='S',help='Seed used to initialise the random number generator.')

        self.parser.add_argument('--model_ids', type=str,nargs="*",help="If this arg is not set, all models are used by default.")
        self.parser.add_argument('--config_paths', type=str,nargs="*",help="If this arg is not set, all config are used by default.")
        self.parser.add_argument('--config_fold', type=str,help="Path to the folder containing the configs. Used only if --config_paths is not set.",default="./configs")
        self.parser.add_argument('--model_args_path', type=str,help="Path to the json containing the arg values that are specific for each model.",default="model_args.json")
        self.parser.add_argument('--output_dir', type=str,help="Path to the directory where the results, models and viz will be saved.",default="../")
        self.parser.add_argument('--data_dir', type=str,help="Path to the directory where the dataset are saved.",default="../data/")

        self.args = None

    def getRemainingArgs(self):
        ''' Reads the comand line arg'''

        self.args = self.parser.parse_args(self.remaining_argv)

def writeConfigFile(args,filePath):
    """ Writes a config file containing all the arguments and their values"""

    config = configparser.ConfigParser()
    config.add_section('default')

    for k, v in  vars(args).items():
        config.set('default', k, str(v))

    with open(filePath, 'w') as f:
        config.write(f)

def addInitArgs(argreader):
    argreader.parser.add_argument('--init_path', type=str, metavar='SM',
                                  help='The path to the weight file to use to initialise the network')
    return argreader

def addValArgs(argreader):

    argreader.parser.add_argument('--metric_early_stop', type=str, metavar='METR',
                                  help='The metric to use to choose the best model')
    argreader.parser.add_argument('--maximise_val_metric', type=str2bool, metavar='BOOL',
                                  help='If true, The chosen metric for chosing the best model will be maximised')
    argreader.parser.add_argument('--max_worse_epoch_nb', type=int, metavar='NB',
                                  help='The number of epochs to wait if the validation performance does not improve.')
    return argreader

def init_post_hoc_arg(argreader):
    argreader.parser.add_argument('--post_hoc_method', type=str, help='The post-hoc method to use instead of the model ')
    argreader.parser.add_argument('--img_nb_per_class', type=int, help='The nb of images on which to compute the att metric.')    
    argreader.parser.add_argument('--noise_tunnel_batchsize', type=int,default=1)
    argreader.parser.add_argument('--expl_batch_size', type=int,default=30)
    argreader.parser.add_argument('--perturbation_maps_resolution', type=str,default="auto") 
    argreader.parser.add_argument('--rise_mask_nb', type=int,default=8000)   
    argreader.parser.add_argument('--multi_step_metrics_step_nb',type=str,default="auto")  
    argreader.parser.add_argument('--load_epoch',type=int)  
    argreader.parser.add_argument('--nt_samples', type=int,default=30)  
    return argreader

def addLossTermArgs(argreader):
    argreader.parser.add_argument('--nll_weight', type=float, metavar='FLOAT',help='The weight of the negative log-likelihood term in the loss function.')
    argreader.parser.add_argument('--focal_weight', type=float, metavar='FLOAT',help='The weight of the focal loss term.')
    argreader.parser.add_argument('--adv_ce_weight', type=float, metavar='FLOAT',help='The weight of the adversarial ce loss term.')   
    argreader.parser.add_argument('--nll_masked_weight', type=float, metavar='FLOAT',help='The weight of the negative log-likelihood term for the masked images.')
    argreader.parser.add_argument('--maximum_pert', type=int, metavar='FLOAT',help='Maximum perturbation when computing adversarial batches. This value should be an integer between 1 and 255.')

    return argreader

def addSalMetrArgs(argreader):
    argreader.parser.add_argument('--sal_metr_bckgr',type=str, help='The filling method to use for saliency metrics. Ignored if --sal_metr_otherimg is True.')
    argreader.parser.add_argument('--sal_metr_mask', type=str2bool, help='To apply the masking of attention metrics during training.')
    argreader.parser.add_argument('--sal_metr_mask_prob', type=float, help='The probability to apply saliency metrics masking.')
    return argreader

def set_debug_mode(args,debug_batch_size=3):
    if os.path.exists("/home/E144069X/"):
        args.debug = True

    if args.debug:
        print("Debug mode.")
        args.val_batch_size = debug_batch_size
        args.first_mod = "resnet18"
        args.img_nb_per_class = 1
        args.cuda = False
        args.perturbation_maps_resolution = 2
        args.rise_mask_nb = 2
        args.multi_step_metrics_step_nb = 7*7
        args.batch_size = debug_batch_size
        args.epochs = 1
        args.debug_batch_size = debug_batch_size
    return args

def is_type(string,tested_type):
    try: 
        tested_type(string)
    except ValueError:
        return False
    else:
        return True

def get_type(value):
    if value in ["True","False"]:
        return value=="True"
    elif is_type(value,int):
        return int(value)
    elif is_type(value,float):
        return float(value)
    else:
        return value

def load_config(config_path):
    config = configparser.ConfigParser()
    config.read(config_path)
    config = dict(config["default"])

    for key in config:
        config[key] = get_type(config[key])

    return argparse.Namespace(**config) 

def get_arg_from_model_config_file(key,config_path):
    config = configparser.ConfigParser()
    config.read(config_path)
    return config["default"][key]

def set_batch_size(args,model_id=None):

    if model_id is None:
        model_id = args.model_id

    if args.cuda and torch.cuda.is_available():
        size = torch.cuda.get_device_properties(0).total_memory/(1024**3)
    else:
        size = 0

    if size<12:
        if "AP" in model_id:
            args.val_batch_size=70
            args.expl_batch_size=30
        else:
            args.val_batch_size=200
            args.expl_batch_size=90
        args.noise_batch_size = 1
    else:
        if "AP" in model_id:
            args.val_batch_size=50
            args.expl_batch_size=40
        else:
            args.val_batch_size=600
            args.expl_batch_size=100
        args.noise_batch_size = 3
    
    return args


