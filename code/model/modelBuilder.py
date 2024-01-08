import torch
from torch import nn
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from . import resnet
from data.load_data import get_class_nb

def buildFeatModel(featModelName, **kwargs):
    if "resnet" in featModelName:
        featModel = getattr(resnet, featModelName)(**kwargs)
    else:
        raise ValueError("Unknown model type : ", featModelName)

    return featModel

class GradCamMod(torch.nn.Module):
    def __init__(self,net):
        super().__init__()
        self.net = net
        self.layer4 = net.firstModel.featMod.layer4
        self.features = net.firstModel.featMod

    def forward(self,x):
        feat = self.net.firstModel.featMod(x)["feat"]

        x = torch.nn.functional.adaptive_avg_pool2d(feat,(1,1))
        x = x.view(x.size(0),-1)
        x = self.net.secondModel.linLay(x)

        return x

# This class is just the class nn.DataParallel that allow running computation on multiple gpus
# but it adds the possibility to access the attribute of the model
class DataParallelModel(nn.DataParallel):
    def __init__(self, model):
        super().__init__(model)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

class Model(nn.Module):

    def __init__(self, firstModel, secondModel):
        super().__init__()
        self.firstModel = firstModel
        self.secondModel = secondModel

    def forward(self, origImgBatch):

        if not self.firstModel is None:

            visResDict = self.firstModel(origImgBatch)

            resDict = self.secondModel(visResDict)

            if visResDict != resDict:
                resDict = merge(visResDict,resDict)

        else:
            resDict = self.secondModel(origImgBatch)

        return resDict

def merge(dictA,dictB,suffix=""):
    for key in dictA.keys():
        if key in dictB:
            dictB[key+"_"+suffix] = dictA[key]
        else:
            dictB[key] = dictA[key]
    return dictB

################################# Visual Model ##########################

class FirstModel(nn.Module):

    def __init__(self, featModelName,**kwargs):
        super().__init__()

        self.featMod = buildFeatModel(featModelName,**kwargs)

    def forward(self, x):
        raise NotImplementedError

class CNN2D(FirstModel):

    def __init__(self, featModelName,**kwargs):
        super().__init__(featModelName,**kwargs)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):

        # N x C x H x L
        self.batchSize = x.size(0)

        # N x C x H x L
        retDict = self.featMod(x)

        if "feat" in retDict:
            retDict["feat_pooled"] = self.avgpool(retDict["feat"]).squeeze(-1).squeeze(-1)

        return retDict

################################ Classification head ########################""

class SecondModel(nn.Module):

    def __init__(self, nbFeat, nbClass):
        super().__init__()
        self.nbFeat, self.nbClass = nbFeat, nbClass

    def forward(self, x):
        raise NotImplementedError

class LinearSecondModel(SecondModel):

    def __init__(self, nbFeat, nbClass, dropout):
        super().__init__(nbFeat, nbClass)
        self.dropout = nn.Dropout(p=dropout)
        self.linLay = nn.Linear(self.nbFeat, self.nbClass,bias=False)

    def forward(self, retDict):
        x = retDict["feat_pooled"]
        x = self.dropout(x)

        output = self.linLay(x)

        retDict["output"]=output

        return retDict

def getResnetFeat(backbone_name, backbone_inplanes):
    if backbone_name in ["resnet50","resnet101","resnet152"]:
        nbFeat = backbone_inplanes * 4 * 2 ** (4 - 1)
    elif backbone_name.find("resnet34") != -1:
        nbFeat = backbone_inplanes * 2 ** (4 - 1)
    elif backbone_name.find("resnet18") != -1:
        nbFeat = backbone_inplanes * 2 ** (4 - 1)
    elif backbone_name == "convnext_small":
        nbFeat = 768
    elif backbone_name == "convnext_base":
        nbFeat = 1024
    elif backbone_name == "vit_b_16":
        nbFeat = 768
    else:
        raise ValueError("Unkown backbone : {}".format(backbone_name))
    return nbFeat

def getResnetDownSampleRatio(args):
    backbone_name = args.first_mod
    if backbone_name.find("resnet") != -1:
        ratio = 32
        for stride in [args.stride_lay2,args.stride_lay3,args.stride_lay4]:
            if stride == 1:
                ratio /= 2

        return int(ratio)

    raise ValueError("Unkown backbone",backbone_name)

def get_feat_map_size(args):
    if "resnet" in args.first_mod:
        resolution = args.img_size//getResnetDownSampleRatio(args)
    else:
        raise ValueError("Can only find feature map size for resnet",args.first_mod)

    return resolution

def netBuilder(args):
    ############### Visual Model #######################

    args.class_nb = get_class_nb(args.data_dir,args.dataset_train)

    nbFeat = getResnetFeat(args.first_mod, args.resnet_chan)

    CNNconst = CNN2D
    if "vit" in args.first_mod:
        kwargs = {"image_size":args.img_size}
    else:
        kwargs = {"chan":args.resnet_chan, "stride":args.resnet_stride,\
                    "strideLay2":args.stride_lay2,"strideLay3":args.stride_lay3,\
                    "strideLay4":args.stride_lay4} 
        
    firstModel = CNNconst(args.first_mod,**kwargs)

    ############### Second Model #######################
    if args.second_mod == "linear":
        secondModel = LinearSecondModel(nbFeat, args.class_nb, args.dropout)
    else:
        raise ValueError("Unknown second model type : ", args.second_mod)

    ############## Whole Model ##########################

    net = Model(firstModel, secondModel)

    net = put_on_gpu(net,args)

    return net

def put_on_gpu(net,args):
    if args.cuda and torch.cuda.is_available():
        net.cuda()
        if args.multi_gpu:
            net = DataParallelModel(net)
    return net 

def addArgs(argreader):
    argreader.parser.add_argument('--first_mod', type=str, metavar='MOD',
                                  help='the net to use to produce feature for each frame')

    argreader.parser.add_argument('--dropout', type=float, metavar='D',
                                  help='The dropout amount on each layer of the RNN except the last one')

    argreader.parser.add_argument('--second_mod', type=str, metavar='MOD',
                                  help='The temporal model. Can be "linear", "lstm" or "score_conv".')

    argreader.parser.add_argument('--resnet_chan', type=int, metavar='INT',
                                  help='The channel number for the visual model when resnet is used')
    argreader.parser.add_argument('--resnet_stride', type=int, metavar='INT',
                                  help='The stride for the visual model when resnet is used')

    argreader.parser.add_argument('--stride_lay2', type=int, metavar='NB',
                                  help='Stride for layer 2.')
    argreader.parser.add_argument('--stride_lay3', type=int, metavar='NB',
                                  help='Stride for layer 3.')
    argreader.parser.add_argument('--stride_lay4', type=int, metavar='NB',
                                  help='Stride for layer 4.')

    return argreader
