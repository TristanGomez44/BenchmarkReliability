

# Enhancing Post-Hoc Explanation Benchmark Reliability for Image Classification

This is the official code repository for this paper: https://arxiv.org/abs/2311.17876

First, install the required package with the environment file:

```
conda env create -f environment.yml
```

Then download the data and format it as described in the next section.

## Data 

Here's how to format the data depending on the dataset you want to use.

### Aircraft

There's nothing to prepare for this dataset, we use the pytorch dataset which downloads and reads the data automatically in the args.data_dir folder.

### CUB 

Download the dataset here: https://data.caltech.edu/records/65de6-vp158
                    
Similarly to the FGVC dataset, format it as follows:

```
python3 data/formatData.py  --cub --data_dir ${DATA_DIR} --archive ${CUB_TAR}
```
where `${CUB_TAR}` is the path to the .tar.gz downloaded.

### Cars 

Currently the cars dataset is not available on its original server (https://ai.stanford.edu/~jkrause/cars/car_dataset.html is down).

However, following the solution proposed here (https://github.com/pytorch/vision/issues/7545), you can still download the dataset as follows:

- Download the images: 'https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset'
- Download the devkit: https://github.com/pytorch/vision/files/11644847/car_devkit.tgz
- Download the 'cars_test_annos_withlabels.mat' file: https://www.kaggle.com/code/subhangaupadhaya/pytorch-stanfordcars-classification/input?select=cars_test_annos_withlabels+%281%29.mat

Now, execute the following command:

```
python3 data/formatData.py  --cars --data_dir ${DATA_DIR} --archive ${IMG_ARCH} --cars_devkit ${DEVKIT} --cars_test_annos ${TEST_ANNOS}
```
where `${IMG_ARCH}`, `${DEVKIT}` and `${TEST_ANNOS}` are the paths to the image archive, the devkit archive and the 'cars_test_annos_withlabels.mat' archive.

Note that the 'cars_test_annos_withlabels.mat' archive only contains the 'cars_test_annos_withlabels.mat' file.

To load the data, we use pytorch dataset, as with the FGVC-Aircraft dataset.

### CROHNIPI

If you want to train on this dataset, send an email to harold.mouchere at univ-nantes.fr.

Once you have the dataset, format it as follows:

```
python3 data/formatData.py --crohn --data_dir ${DATA_DIR} --archive ${CROHN_DIR}
```
where `${CROHN_DIR}` is the path to the zip containing the dataset.

### Embryo 

Download the 'embryo_dataset.tar.gz' and the 'embryo_dataset_annotations.tar.gz' here : https://doi.org/10.5281/zenodo.6390798

Construct the embryo image dataset using the following command : 

```
python3 data/formatData.py --embryo --data_dir ${DATA_DIR} --archive ${EMBRYO_DATASET_TAR} --embryo_annotations_root ${EMBRYO_DATASET_ANNOT_TAR}
```

where `${DATA_DIR}`, `${EMBRYO_DATASET_TAR}` and `${EMBRYO_DATASET_ANNOT_TAR}` correspond to the folder where you want the datasets to be, the 'embryo_dataset.tar.gz' archive and the 'embryo_dataset_annotations.tar.gz' archive.

## Scripts 

Now you have downloaded the data you can execute the scripts to reproduce our experiments.

First go in the 'code' folder of the project and train the models:

```
python3 train_test.py --config_paths ${CONFIG_PATHS}
```
where `${CONFIG_PATHS}` are the paths to the config you want to train with.
Each config correspond to one dataset.
If you want to train on the cub dataset, run:
```
python3 train_test.py --config_paths configs/model_cub.config
```
The results and model weights will be saved at the path set by the args.output_dir, which by default is "../".
So, by default, the script will create a "models" and a "results" folder at the root of the project (because you need to execute the scripts in the code folder).
In the results folder, you will find several folder, one for each dataset.
With the above example, there will be a "CUB" folder containing csv indicating the performance of the trained models.

Once the model is trained, compute the explanations and evaluate them with the faithfulness metrics:
```
python3 compute_and_evaluate_explanations.py --config_paths ${CONFIG_PATHS}
```
If you used the CUB dataset, this will create a database at the path `{args.output_dir}/results/{EXP_ID}`/saliency_metrics.db which contains the values of the faithfulness metrics, where `${EXP_ID}` will be 'CUB' if you used the CUB dataset, 'AIR' for the FGVC-Aircraft dataset etc.

Now, compute krippendorff's alpha:
```
python3 krippendorf_alpha.py --config_paths ${CONFIG_PATHS}
```
This script uses the database computed previously to compute Krippendorff's alpha.
The results will be stored in csv in the same folder than the previous script.
For example, you will find a latex table with the average Krippendorff alpha's for each training setting named "krippendorff_mean_diff_Baseline.tex".
The script also generates visualization that you will find at `{args.output_dir}/vis/{EXP_ID}/krippendorff_mean_diff_Baseline.png`

If you also want to compute how much can the benchmark test size be reduced:
```
python3 test_size_reduction.py --config_paths ${CONFIG_PATHS}
```
The results are stored in the `{args.output_dir}/results/{EXP_ID}`

Note that by default, each script runs on all models.
This means that when you run the command below, all the 8 models will be trained sucessively:
```
python3 train_test.py --config_paths configs/model_cub.config
```

If you want to train only on some of the settings, let's say the Baseline and the FP+FL settings, use the '--model_ids' argument:
```
python3 train_test.py --config_paths configs/model_cub.config --model_ids Baseline FP+FL
```
