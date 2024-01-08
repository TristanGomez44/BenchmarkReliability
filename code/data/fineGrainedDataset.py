""" CUB-200-2011 (Bird) Dataset
Created: Oct 11,2019 - Yuchong Gu
Revised: Oct 11,2019 - Yuchong Gu
"""
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
import torch
import torchvision
from . import formatData

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

def get_dataset(args,phase,img_size):

    dataset_name = getattr(args,"dataset_"+phase)

    is_color_dataset = "emb" not in dataset_name
    transf = get_transform(img_size, phase,colorDataset=is_color_dataset)

    if dataset_name.startswith("aircraft"):
        dataset = FGVCAircraftCustom(args.data_dir,split=phase,annotation_level="variant",transform=transf,download=True)
    elif dataset_name.startswith("cars"):
        if phase == "val":
            phase = "train"
        dataset = StanfordCarsCustom(args.data_dir,split=phase,transform=transf,download=True)
    else:
        dataset = FineGrainedDataset(os.path.join(args.data_dir,dataset_name), phase,(img_size,img_size),transform=transf)
    
    return dataset

class FGVCAircraftCustom(torchvision.datasets.FGVCAircraft):
    def __init__(self,root,split,annotation_level,transform,download):
        super().__init__(root,split,annotation_level,transform,download=download)
        self.image_label = {i:self._labels[i] for i in range(len(self._labels))}

class StanfordCarsCustom(torchvision.datasets.StanfordCars):

    def __init__(self,root,split,transform,download):
        super().__init__(root,split=split,transform=transform,download=download)
        self.image_label = {i:self._samples[i][1] for i in range(len(self._samples))}
        self.num_classes = len(set(self.image_label.values()))
        print("CUSTOM CARS",self.num_classes)

class FineGrainedDataset(Dataset):
    """
    # Description:
        Dataset for retrieving CUB-200-2011 images and labels
    # Member Functions:
        __init__(self, phase, resize):  initializes a dataset
            phase:                      a string in ['train', 'val', 'test']
            resize:                     output shape/size of an image
        __getitem__(self, item):        returns an image
            item:                       the idex of image in the whole dataset
        __len__(self):                  returns the length of dataset
    """

    def __init__(self, root, phase,resize,transform):

        self.image_path = {}
        self.image_label = {}
        self.root = root
        self.phase = phase
        self.resize = resize
        self.image_id = []
        self.classes = [d.name for d in os.scandir(self.root) if d.is_dir()]

        self.num_classes = len(self.classes)

        if root.find("emb") != -1:
            self.classes.sort(key = lambda x:formatData.labelDict[x])
        else:
            self.classes.sort()

        class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        instances = []
        id = 0
        self.class_to_img_ids = {class_ind:[] for class_ind in np.arange(self.num_classes)}
        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            target_dir = os.path.join(self.root, target_class)
            if not os.path.isdir(target_dir):
                continue
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    if is_valid_file(path):
                        item = path, class_index
                        instances.append(item)

                        self.image_path[id] = path
                        self.image_label[id] = class_index
                        self.image_id.append(id)

                        self.class_to_img_ids[class_index].append(id)

                        id += 1

        self.is_color_dataset = "emb" not in root

        # transform
        self.transform = transform

    def get_candidate_img_list(self,candidate_class_labels):
        rand_ind = np.random.randint(0,len(candidate_class_labels),size=(1,))[0]
        class_to_sample_from = candidate_class_labels[rand_ind] 
        candidate_img_list = self.class_to_img_ids[class_to_sample_from]
        return candidate_img_list,class_to_sample_from

    def __getitem__(self, item):
        image_id = self.image_id[item]
        image = Image.open(self.image_path[image_id]).convert('RGB')  # (C, H, W)
        image = self.transform(image)
        return image, self.image_label[image_id]

    def __len__(self):
        return len(self.image_id)

def is_valid_file(x):
    return has_file_allowed_extension(x, IMG_EXTENSIONS)

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)
    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)

def add_patches_func(img,patch_res):

    cent_x,cent_y = torch.randint(0,patch_res,size=(2,))
    var_x,var_y = torch.rand(size=(2,))+1

    x = torch.arange(patch_res).unsqueeze(0)
    y = torch.arange(patch_res).unsqueeze(1)

    mask = torch.exp(-((x-cent_x)**2)/var_x-((y-cent_y)**2)/var_y)
    mask = (mask - mask.min())/(mask.max() - mask.min())
    mask = mask.unsqueeze(0).unsqueeze(0)

    if torch.rand(size=(1,)) > 0.5:
        mask = (mask > 0.5)*1.0
        mask = torch.nn.functional.interpolate(mask,img.shape[1:],mode="bicubic",align_corners=False).clamp(min=0, max=1)[0]
    else:
        k = torch.randint(0,patch_res*patch_res,size=(1,)).item()
        values,_ = torch.topk(mask.view(-1),k,0,sorted=True)
        mask = (mask > values[-1]) * 1.0
        mask = torch.nn.functional.interpolate(mask,img.shape[1:],mode="nearest")[0]

    return mask

def get_imgnet_mean_std():
    return {"mean":[0.485, 0.456, 0.406],"std":[0.229, 0.224, 0.225]}

def get_transform(resize, phase='train',colorDataset=True,sqResizing=True,cropRatio=0.875,brightness=0.126,saturation=0.5):

    if sqResizing:
        kwargs={"size":(int(resize[0] / cropRatio), int(resize[1] / cropRatio))}
    else:
        kwargs={"size":int(resize[0] / cropRatio)}

    if phase == 'train':
        transf = [transforms.Resize(**kwargs),
                    transforms.RandomCrop(resize),
                    transforms.RandomHorizontalFlip(0.5)]

        if colorDataset:
            transf.extend([transforms.ColorJitter(brightness=brightness, saturation=saturation)])

    else:
        transf = [transforms.Resize(**kwargs),transforms.CenterCrop(resize)]

    transf.extend([transforms.ToTensor()])

    if colorDataset:
        mean_std_dict = get_imgnet_mean_std()
        mean,std = mean_std_dict["mean"],mean_std_dict["std"]
        transf.extend([transforms.Normalize(mean=mean, std=std)])

    transf = transforms.Compose(transf)

    return transf