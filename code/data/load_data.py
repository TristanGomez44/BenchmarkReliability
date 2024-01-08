import glob,os
import numpy as np
import torch
import args
from . import fineGrainedDataset

def make_label_list(dataset):
    return [dataset.image_label[img_ind] for img_ind in sorted(dataset.image_label.keys())]

def sample_img_inds(nb_per_class,label_list=None,testDataset=None):

    assert (label_list is not None) or (testDataset is not None)
    
    if label_list is None:
        label_list = make_label_list(testDataset)

    label_to_ind = {}
    for i in range(len(label_list)):
        lab = label_list[i]

        if type(lab) is torch.Tensor:
            lab = lab.item()

        if lab not in label_to_ind:
            label_to_ind[lab] = []  
        label_to_ind[lab].append(i)

    torch.manual_seed(0)
    chosen_inds = []
    for label in sorted(label_to_ind):
        all_inds = torch.tensor(label_to_ind[label])
        all_inds_perm = all_inds[torch.randperm(len(all_inds))]
        chosen_class_inds = all_inds_perm[:nb_per_class]
        if len(chosen_class_inds) < nb_per_class:
            raise ValueError(f"Number of image to be sampled per class is too high for class {label} which has only {len(chosen_class_inds)} images.")
        chosen_inds.extend(chosen_class_inds)
    
    chosen_inds = torch.tensor(chosen_inds)

    return chosen_inds 

def get_class_nb(data_dir,dataset_train):
    if dataset_train.startswith("aircraft"):
        args.class_nb = 100 
    elif dataset_train.startswith("cars"):
        args.class_nb = 196
    else:
        args.class_nb = len(glob.glob(os.path.join(f"{data_dir}/",dataset_train,"*/")))
        if args.class_nb == 0:
            raise ValueError("Found 0 classes in",os.path.join(dataset_train,"*/"))
    return args.class_nb 
    
def buildTrainLoader(args):

    imgSize = args.img_size

    train_dataset = fineGrainedDataset.get_dataset(args,"train",(imgSize,imgSize))

    totalLength = len(train_dataset)

    train_prop = args.train_prop

    np.random.seed(1)
    torch.manual_seed(1)
    if args.cuda:
        torch.cuda.manual_seed(1)
    
    if args.dataset_train == args.dataset_val:
        train_dataset, _ = torch.utils.data.random_split(train_dataset, [int(totalLength * train_prop),totalLength - int(totalLength * train_prop)])

        train_dataset.num_classes = train_dataset.dataset.num_classes
        train_dataset.image_label = train_dataset.dataset.image_label

    bsz = args.batch_size

    trainLoader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=bsz,pin_memory=True, num_workers=args.num_workers, shuffle=True)

    return trainLoader, train_dataset

def buildTestLoader(args, mode):
    imgSize = args.img_size
    test_dataset = fineGrainedDataset.get_dataset(args,mode,(imgSize,imgSize))

    if mode == "val" and args.dataset_train == args.dataset_val:
        np.random.seed(1)
        torch.manual_seed(1)
        if args.cuda:
            torch.cuda.manual_seed(1)

        train_prop = args.train_prop

        totalLength = len(test_dataset)
        _, test_dataset = torch.utils.data.random_split(test_dataset, [int(totalLength * train_prop),totalLength - int(totalLength * train_prop)])

        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)

    testLoader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.val_batch_size,num_workers=args.num_workers,pin_memory=True)

    return testLoader,test_dataset

def getBatch(testDataset,inds,args):
    data_list = []
    targ_list = []
    for i in inds:
        batch = testDataset.__getitem__(i)
        data,targ = batch[0].unsqueeze(0),torch.tensor(batch[1]).unsqueeze(0)
        data_list.append(data)
        targ_list.append(targ)
    
    data_list = torch.cat(data_list,dim=0)
    targ_list = torch.cat(targ_list,dim=0)

    if args.cuda:
        data_list = data_list.cuda() 
        targ_list = targ_list.cuda()
        
    return data_list,targ_list 

def addArgs(argreader):

    argreader.parser.add_argument('--batch_size', type=int, metavar='BS',
                                  help='The batchsize to use for training')
    argreader.parser.add_argument('--val_batch_size', type=int, metavar='BS',
                                  help='The batchsize to use for validation')

    argreader.parser.add_argument('--train_prop', type=float, metavar='END',
                                  help='The proportion of the train dataset to use for training when working in non video mode. The rest will be used for validation.')

    argreader.parser.add_argument('--dataset_train', type=str, metavar='DATASET',
                                  help='The dataset for training. Can be "big" or "small"')
    argreader.parser.add_argument('--dataset_val', type=str, metavar='DATASET',
                                  help='The dataset for validation. Can be "big" or "small"')
    argreader.parser.add_argument('--dataset_test', type=str, metavar='DATASET',
                                  help='The dataset for testing. Can be "big" or "small"')

    return argreader
