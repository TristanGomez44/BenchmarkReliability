import os
import glob
from shutil import copyfile
import subprocess

import numpy as np
import torch

from args import str2bool,addInitArgs,addValArgs,init_post_hoc_arg,addLossTermArgs,addSalMetrArgs,writeConfigFile,set_debug_mode
from data import load_data
from metrics import training_metrics
from model import init_model,modelBuilder
from training import sal_metr_data_aug,update,multi_obj_epoch_selection
from training.loss import Loss,agregate_losses
from utils import getEpoch

def training_epoch(model, optim, loader, epoch, args, **kwargs):

    model.train()

    print("Epoch", epoch, " : train")

    metrDict = None
    validBatch = 0
    totalImgNb = 0

    var_dic = {}
    for batch_idx, batch in enumerate(loader):
        optim.zero_grad()

        if batch_idx % args.log_interval == 0:
            processedImgNb = batch_idx * len(batch[0])
            print("\t", processedImgNb, "/", len(loader.dataset))

        data, target = batch[0], batch[1]
        
        if args.cuda:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)

        resDict = model(data)

        resDict,_,data_masked = sal_metr_data_aug.apply_sal_metr_masks_and_update_dic(model,data,args,resDict)

        output = resDict["output"]
        if args.adv_ce_weight > 0:
            data_adv = kwargs["lossFunc"].atk(data, target)
            resDict["output_adv"] = model(data_adv)["output"]
            data_masked_adv = kwargs["lossFunc"].atk(data_masked, target)
            resDict["output_masked_adv"] = model(data_masked_adv)["output"]

        loss_dic = kwargs["lossFunc"](output, target, resDict)
        loss_dic = agregate_losses(loss_dic)
        loss = loss_dic["loss"]/len(data)
        loss.backward()

        optim.step()
        optim.zero_grad()

        # Metrics
        metDictSample = training_metrics.binaryToMetrics(output, target,resDict)
        metDictSample = training_metrics.add_losses_to_dic(metDictSample,loss_dic)
        metrDict = training_metrics.updateMetrDict(metrDict, metDictSample)

        var_dic = update.all_cat_var_dic(var_dic,resDict,target)
            
        validBatch += 1
        totalImgNb += len(data)

        if args.debug:
            break

    print(f"{args.output_dir}/models/{args.exp_id}/model{args.model_id}_epoch{epoch}")
    torch.save(model.state_dict(), f"{args.output_dir}/models/{args.exp_id}/model{args.model_id}_epoch{epoch}")
    
    metrDict = training_metrics.expected_calibration_error(var_dic, metrDict)

    writeSummaries(metrDict, totalImgNb, epoch, "train", args.model_id, args.exp_id,args.output_dir)

    return metrDict

def evaluation(model, loader, epoch, args, mode="val",**kwargs):
    torch.set_grad_enabled(False)

    model.eval()

    print("Epoch", epoch, " : {}".format(mode))

    metrDict = None

    validBatch = 0
    totalImgNb = 0
    var_dic = {}
    for batch_idx, batch in enumerate(loader):
        data, target = batch[:2]

        if (batch_idx % args.log_interval == 0):
            print("\t", batch_idx * len(data), "/", len(loader.dataset))

        if args.cuda: data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
      
        resDict = model(data)

        resDict,_,data_masked= sal_metr_data_aug.apply_sal_metr_masks_and_update_dic(model,data,args,resDict)

        output = resDict["output"]

        if args.adv_ce_weight > 0:
            torch.set_grad_enabled(True)
            data_adv = kwargs["lossFunc"].atk(data, target)
            if args.loss_on_masked:
                data_masked_adv = kwargs["lossFunc"].atk(data_masked, target)

            model.eval()
            torch.set_grad_enabled(False)  
            resDict["output_adv"] = model(data_adv)["output"]
            if args.loss_on_masked:
                resDict["output_masked_adv"] = model(data_masked_adv)["output"]

        loss_dic = kwargs["lossFunc"](output, target, resDict)
        loss_dic = agregate_losses(loss_dic)

        # Metrics
        metDictSample = training_metrics.binaryToMetrics(output, target,resDict)
        metDictSample = training_metrics.add_losses_to_dic(metDictSample,loss_dic)
        metrDict = training_metrics.updateMetrDict(metrDict, metDictSample)
        var_dic = update.all_cat_var_dic(var_dic,resDict,target)

        validBatch += 1
        totalImgNb += len(data)

        if args.debug:
            break

    metrDict = training_metrics.expected_calibration_error(var_dic, metrDict)
    
    writeSummaries(metrDict, totalImgNb, epoch, mode, args.model_id, args.exp_id,args.output_dir)

    torch.set_grad_enabled(True)

    return metrDict["Accuracy"]

def writeSummaries(metrDict, totalImgNb, epoch, mode, model_id, exp_id,output_dir):
 
    for metric in metrDict.keys():
        if metric != "temperature" and metric.find("_val_rate") == -1 and metric.find("ECE") == -1:
            metrDict[metric] /= totalImgNb

    header_list = ["epoch"]
    header_list += [metric.lower().replace(" ", "_") for metric in metrDict.keys()]
    header = ",".join(header_list)

    csv_path = f"{output_dir}/results/{exp_id}/metrics_{model_id}_{mode}.csv"

    if not os.path.exists(csv_path):
        with open(csv_path, "w") as text_file:
           print(header, file=text_file) 

    with open(csv_path, "a") as text_file:
        print(epoch,file=text_file,end=",")
        print(",".join([str(metrDict[metric]) for metric in metrDict.keys()]), file=text_file)

    return metrDict

def addOptimArgs(argreader):
    argreader.parser.add_argument('--lr', type=float, metavar='LR',
                                  help='learning rate')
    argreader.parser.add_argument('--momentum', type=float, metavar='M',
                                  help='SGD momentum')
    argreader.parser.add_argument('--weight_decay', type=float, metavar='M',
                                  help='Weight decay')
    argreader.parser.add_argument('--optim', type=str, metavar='OPTIM',
                                  help='the optimizer to use (default: \'SGD\')')
    return argreader

def addAllTrainingArgs(argreader):
    argreader.parser.add_argument('--loss_on_masked',type=str2bool, help='To apply the focal loss on the output corresponding to masked data.')
    argreader.parser.add_argument('--multi_obj_sel', type=str2bool, metavar='S')

    argreader = addInitArgs(argreader)
    argreader = addOptimArgs(argreader)
    argreader = addValArgs(argreader)
    argreader = addLossTermArgs(argreader)
    argreader = addSalMetrArgs(argreader)
    argreader = init_post_hoc_arg(argreader)

    argreader = modelBuilder.addArgs(argreader)
    argreader = load_data.addArgs(argreader)

    return argreader

def save_git_status(args):

    path_start = f"{args.output_dir}/models/{args.exp_id}/model{args.model_id}"

    cmd_list = ["git status","git diff","git rev-parse --short HEAD"]
    labels = ["status","diff","commit"]

    for labels,cmd in zip(labels,cmd_list):
        path = path_start + "_git_"+labels+".txt"
        output = subprocess.check_output(cmd.split(" "),text=True)
        with open(path,"w") as file:
            print(output,file=file)

def train(args):

    args = set_debug_mode(args)   
    args.cuda = args.cuda and torch.cuda.is_available()

    for fold_name in ["vis","results","models"]:
        os.makedirs(f"{args.output_dir}/{fold_name}/{args.exp_id}",exist_ok=True)

    writeConfigFile(args,f"{args.output_dir}/models/{args.exp_id}/{args.model_id}.ini")
    print("Model :", args.model_id, "Experience :", args.exp_id)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    args.class_nb = load_data.get_class_nb(args.data_dir,args.dataset_train)

    save_git_status(args)

    trainLoader,_ = load_data.buildTrainLoader(args)
    valLoader,_ = load_data.buildTestLoader(args,"val")
    
    # Building the net
    net = modelBuilder.netBuilder(args)

    kwargsTr = {'loader': trainLoader, 'args': args}
    kwargsVal = kwargsTr.copy()

    kwargsVal['loader'] = valLoader

    startEpoch = 1
    epoch = startEpoch
    bestEpoch = epoch
    worseEpochNb = 0

    kwargsTr["optim"] = init_model.getOptim(args.optim, args.lr,args.momentum,args.weight_decay,net)

    bestMetricVal = -np.inf
    isBetter = lambda x, y: x > y

    lossFunc = Loss(args,reduction="sum",net=net)

    if args.multi_gpu:
        lossFunc = torch.nn.DataParallel(lossFunc)
        if args.adv_ce_weight > 0:
            lossFunc.atk = lossFunc.module.atk

    kwargsTr["lossFunc"],kwargsVal["lossFunc"] = lossFunc,lossFunc

    while epoch < args.epochs + 1 and worseEpochNb < args.max_worse_epoch_nb:
        
        kwargsTr["epoch"], kwargsVal["epoch"] = epoch, epoch
        kwargsTr["model"], kwargsVal["model"] = net, net

        training_epoch(**kwargsTr)

        metricVal = evaluation(**kwargsVal)

        bestEpoch, bestMetricVal, worseEpochNb = update.updateBestModel(metricVal, bestMetricVal, args.exp_id,args.model_id, bestEpoch, epoch, net,isBetter,worseEpochNb,output_dir=args.output_dir)

        epoch += 1

    kwargsTest = kwargsVal
    kwargsTest["mode"] = "test"

    testLoader,_ = load_data.buildTestLoader(args, "test")

    kwargsTest['loader'] = testLoader

    current_best_weights_path = f"{args.output_dir}/models/{args.exp_id}/model{args.model_id}_best_epoch{bestEpoch}"

    val_metrics_path = f"{args.output_dir}/results/{args.exp_id}/metrics_{args.model_id}_val.csv"
    multi_obj_best_epoch = multi_obj_epoch_selection.acc_and_ece_selection(val_metrics_path)
    multi_obj_weights_path = f"{args.output_dir}/models/{args.exp_id}/model{args.model_id}_epoch{multi_obj_best_epoch}"

    if multi_obj_best_epoch != bestEpoch:

        new_best_weights_path = multi_obj_weights_path.replace("epoch","best_epoch")

        #Saves best epoch weights
        if not os.path.exists(new_best_weights_path):
            if os.path.exists(current_best_weights_path):
                copyfile(current_best_weights_path,new_best_weights_path)
            else:
                raise ValueError(current_best_weights_path+" is missing")

        #Remove best path 
        if os.path.exists(current_best_weights_path):
            os.remove(current_best_weights_path)

        current_best_weights_path = new_best_weights_path

        bestEpoch = multi_obj_best_epoch

    #Now that the best weight is saved, we remove all the weights saved during training to free space except the last one 
    file_paths = glob.glob(f"{args.output_dir}/models/{args.exp_id}/model{args.model_id}_epoch*")
    file_paths = sorted(file_paths,key=getEpoch)
    file_paths = file_paths[:-1]

    for path in file_paths:
        if os.path.basename(path).startswith(f"model{args.model_id}_epoch"):
            os.remove(path)

    weights_path = current_best_weights_path

    net = init_model.preprocessAndLoadParams(weights_path,args.cuda,net)

    kwargsTest["model"] = net
    kwargsTest["epoch"] = bestEpoch

    evaluation(**kwargsTest)