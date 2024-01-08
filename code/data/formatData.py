import argparse
import random
import glob
import os
from os.path import join

from pathlib import Path
import numpy as np
import cv2
from shutil import copyfile
from scipy.io import loadmat
import subprocess
import shutil
import tarfile
import zipfile

labelDict = {"tPB2":0,"tPNa":1,"tPNf":2,"t2":3,"t3":4,"t4":5,"t5":6,"t6":7,"t7":8,"t8":9,"t9+":10,"tM":11,"tSB":12,"tB":13,"tEB":14,"tHB":15}

def getLabDic():
	return labelDict

def getRevLab():
	rev = {}
	for key in labelDict:
		rev[labelDict[key]] = key
	return rev

def extract(path):

	if os.path.splitext(path)[1] == ".zip":
		with zipfile.ZipFile(path, 'r') as zip_ref:
			zip_ref.extractall(os.path.dirname(path))
		path = path.replace(".zip","")
		
	else:
		tar = tarfile.open(path)
		tar.extractall(os.path.dirname(path))
		path = path.replace(".tar.gz","").replace(".tgz","")

	return path

def extract_if_not_already_done(path):
	ext = os.path.splitext(path)[1]
	root = path.replace(ext,"")
	print(root)
	if not os.path.exists(root):
		extract(path)	
	return root

def formatEmbryo(path,annotations_path,dataset_prefix,train_prop,val_prop,proportion_of_videos_to_extract,data_dir,debug):

	tar_root = tarfile.open(path)
	tar_root.extractall(os.path.dirname(path))

	root = extract_if_not_already_done(path)
	annotations_root = extract_if_not_already_done(annotations_path)

	annotations_root = annotations_root.replace(".tar.gz","/")

	allPaths = sorted(glob.glob(root+"/*/"))

	random.seed(0)
	random.shuffle(allPaths)

	nbTrain = int(len(allPaths)*train_prop)
	nbVal = int(len(allPaths)*val_prop)
	
	trainPaths = allPaths[:nbTrain]
	valPaths = allPaths[nbTrain:nbTrain+nbVal]
	testPaths = allPaths[nbTrain+nbVal:]

	for subset_name,subset_paths in zip(["train","val","test"],[trainPaths,valPaths,testPaths]):
		makeEmbryoSubset(annotations_root,subset_paths,subset_name,dataset_prefix,proportion_of_videos_to_extract,data_dir,debug)

def makeEmbryoSubset(annotations_root,paths,subset,dataset_prefix,prop,data_dir,debug):

	subset_path = join(data_dir,dataset_prefix+"_"+subset+"/")
	if not os.path.exists(subset_path):
		os.makedirs(subset_path)

	for label in labelDict.keys():
		label_fold = subset_path+label+"/"
		if not os.path.exists(label_fold):
			os.makedirs(label_fold)
	
	print(subset)
	
	for k,path in enumerate(paths):

		vidName =path.split("/")[-2]
		print("\t",vidName,k,"/",len(paths))
		frame_paths = glob.glob(join(path,"*.jpeg"))
		
		if len(frame_paths) == 0:
			frame_paths = glob.glob(join(path,"F0","*.jpeg"))

		annot_path = f"{annotations_root}/{vidName}_phases.csv"

		annot = np.genfromtxt(annot_path,delimiter=",",dtype="str")

		for i,frame_path in enumerate(frame_paths):
			frame = cv2.imread(frame_path)
			frame_ind = int(os.path.splitext(os.path.basename(frame_path))[0].split("RUN")[1])

			#Removing top border
			if frame.shape[0] > frame.shape[1]:
				frame = frame[:,frame.shape[0]-frame.shape[1]:]

			#Removing time annotation on the bottom of the image
			frame[-30:] = 0

			if i%(1/prop) == 0:
				label = getClass(annot,frame_ind)

				if label is not None:
					cv2.imwrite(join(subset_path,label,vidName+"_"+str(frame_ind)+".png"),frame)

		if debug:
			break

def getClass(annot,i):
	for row in annot:

		if int(row[1])<=i and i<=int(row[2]):
			return row[0]

	return None


def formatCUB(path,data_dir,prefix="CUB_200_2011"):

	root = extract_if_not_already_done(path)

	print(root,os.path.split(root)[0])

	root = join(os.path.split(root)[0],prefix)

	for subset in ["train","test"]:
		subset_path = join(data_dir,"data",f"{prefix}_{subset}")
		if not os.path.exists(subset_path):
			os.makedirs(subset_path)

	is_train_list = np.genfromtxt(join(root,"train_test_split.txt"),delimiter=" ")[:,1]
	labels = np.genfromtxt(join(root,"image_class_labels.txt"),delimiter=" ",dtype=str)[:,1]
	paths = np.genfromtxt(join(root,"images.txt"),delimiter=" ",dtype=str)[:,1]

	for i in range(len(is_train_list)):

		if i%100 == 0:
			print(i)

		path,label,is_train = paths[i],labels[i],is_train_list[i]
		folder = f"{prefix}_train" if is_train else f"{prefix}_test"
		filename = os.path.basename(path)

		dest_folder = join(data_dir,folder,label)

		if not os.path.exists(dest_folder):
			os.makedirs(dest_folder)

		source_path = join(root,"images",path)
		
		copyfile(source_path,join(dest_folder,filename))

def formatCars(imgs_path,devkit_path,test_annos_path,data_dir):

	for path in [imgs_path,devkit_path,test_annos_path]:
		print(path)
		extract_if_not_already_done(path)

	root = join(data_dir,"stanford_cars")
	os.makedirs(root,exist_ok=True)

	for subset in ["train","test"]:
		source = join(data_dir,"cars_"+subset,"cars_"+subset)
		dest = join(data_dir,"cars_"+subset)

		img_paths = glob.glob(join(source,"*.jpg"))
		for img_path in img_paths:
			shutil.move(img_path,dest)

		if os.path.exists(source):
			shutil.rmtree(source)

		if not os.path.exists(join(root,"cars_"+subset)):
			shutil.move(dest,root)

	if not os.path.exists(join(root,"devkit")):
		shutil.move(join(os.path.dirname(devkit_path),"devkit"),root)
	
	new_filename = "cars_test_annos_withlabels (1).mat"
	annos_path = join(os.path.dirname(test_annos_path),new_filename)
	if not os.path.exists(join(root,new_filename)):
		shutil.move(annos_path,join(root,new_filename))

def formatCrohn(path,data_dir,prefix="DataCrohnIPI"):
	
	extract_if_not_already_done(path)
	root = join(os.path.dirname(path),prefix)

	csv_path = join(root,"CrohnIPI_description.csv")

	arr = np.genfromtxt(csv_path,dtype=str,delimiter=",")[1:]

	#Removing the '>' character
	arr[:,1] = np.array(list(map(lambda x:x.replace(">",""),arr[:,1])))

	fold_nb = 5 #The number of split is supposed to be 5
	classes = list(set(arr[:,1]))

	class_dic = {row[0]:{"class":row[1],"split":int(row[2])} for row in arr}

	for fold_ind in range(1,fold_nb+1):
		for data_set in ["train","val","test"]:
			os.makedirs(f"{data_dir}/crohn_{fold_ind}_{data_set}/",exist_ok=True)

			for class_name in classes:
				os.makedirs(f"{data_dir}/crohn_{fold_ind}_{data_set}/{class_name}/",exist_ok=True)

	splits = np.arange(fold_nb)

	for i,frame_name in enumerate(class_dic):
		class_name,split = class_dic[frame_name]["class"],class_dic[frame_name]["split"]

		source = join(root,"imgs",frame_name)

		for fold_ind in range(1,fold_nb+1):

			splits_shifted = ((splits + (fold_ind - 1)) % fold_nb) + 1

			if split in list(splits_shifted[:3]):
				data_set = "train"
			elif split == splits_shifted[3]:
				data_set = "val"
			else:
				data_set = "test"

			destination = f"{data_dir}/crohn_{fold_ind}_{data_set}/{class_name}/{frame_name}"

			shutil.copy(source,destination)


def main():

	#Getting arguments from config file and command row
	#Building the arg reader
	parser = argparse.ArgumentParser()

	parser.add_argument('--embryo',action="store_true",help='To format the embryo dataset')
	parser.add_argument('--cars',action="store_true",help='To format the cars dataset')
	parser.add_argument('--cub',action="store_true",help='To format the CUB dataset')
	parser.add_argument('--crohn',action="store_true",help='To format the CROHN dataset')

	parser.add_argument('--data_dir',type=str,help="Folder where the datasets will be stored.",default="../data")
	parser.add_argument('--debug',action="store_true")
	parser.add_argument('--archive',type=str,help="path to the archive of the dataset")
	
	################### Embryo dataset args ###################"
	parser.add_argument('--embryo_annotations_root',type=str,default="../data/embryo_dataset_annotations.tar.gz")
	parser.add_argument('--dataset_prefix',type=str,default="embryo_img_fixed")
	parser.add_argument('--train_prop',type=float,default=0.4)
	parser.add_argument('--val_prop',type=float,default=0.1)
	parser.add_argument('--proportion_of_videos_to_extract',type=float,default=1/5)

	################### CARS Args ####################
	parser.add_argument('--cars_devkit',type=str)
	parser.add_argument('--cars_test_annos',type=str)

	args = parser.parse_args()

	if args.embryo:
		formatEmbryo(args.archive,args.embryo_annotations_root,args.dataset_prefix,args.train_prop,args.val_prop,args.proportion_of_videos_to_extract,args.data_dir,args.debug)
	if args.cub:
		formatCUB(args.archive,args.data_dir)
	if args.cars:
		formatCars(args.archive,args.cars_devkit,args.cars_test_annos,args.data_dir)
	if args.crohn:
		formatCrohn(args.archive,args.data_dir)


if __name__ == "__main__":
	main()
