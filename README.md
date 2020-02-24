# NeurIPS 2019: Disentanglement Challenge

This is the code release for our submissions to the [NeurIPS 2019 Disentanglement Challenge](https://www.aicrowd.com/challenges/neurips-2019-disentanglement-challenge). 
We achieved the second place in both stage 1 and stage 2 of the challenge.

You can find the accompanying reports on Arxiv soon. 
If you make use of our ideas or this codebase in your research, please consider [citing the reports](#citing).

This codebase builds heavily on AIcrowd's [Disentanglement Challenge Starter Kit](https://github.com/AIcrowd/neurips2019_disentanglement_challenge_starter_kit). 
The repository looks a bit complicated as it includes many files only necessary to work with the challenge evaluation system. 
It is still possible to execute and evaluate the code fully locally on your machine.

### Setup

- **Anaconda** (By following instructions [here](https://www.anaconda.com/download)) At least version `4.5.11` is required to correctly populate `environment.yml`.

* **Create the conda environment from `environment.yml`**

```sh
conda env create -f environment.yml --name disentanglement_challenge
conda activate disentanglement_challenge
```

### Download and Prepare Datasets

Follow the instructions [here](https://github.com/google-research/disentanglement_lib#downloading-the-data-sets) to download the publicly available datasets, and store them inside `./scratch/datasets` folder.
The `mpi3d_realistic` and `mpi3d_real` datasets can be obtained by following the instructions [here](https://github.com/rr-learning/disentanglement_dataset).

### Train and Test

You can run an experiment using the `run.sh` script.

```
./run.sh <path_to_config> <experiment_name>
```

Example for our stage 1 submission:

```
./run.sh configs/stage1.gin stage1
```

This will first perform training, and then do all necessary steps to evaluate the performance using the [disentanglement_lib](https://github.com/google-research/disentanglement_lib).

#### Finetuning the Feature Extractor (Stage 2 Submission)

For our stage 2 submission, it is first necessary to finetune the feature extractor in a separate step before training the VAE. 
You can do so by running:

```
source train_environ.sh
python pytorch/train_extractor configs/stage2_finetuning.gin stage2_finetuning
```

This requires both `mpi3d_toy` and `mpi3d_realistic` to be downloaded.
After finetuning is finished, you need to move the the resulting checkpoint:

```
mv scratch/shared/stage2_finetuning/AggVGG19_64x64.pth torch_home
```

Now you can run the main training step for stage 2:

```
./run.sh configs/stage2_vae.gin stage2
```

#### Settings Tweaks

- **Changing the dataset:** You can adapt the dataset that is used by changing the line `export AICROWD_DATASET_NAME=mpi3d_toy` in `train_environ.sh`.
- **Changing the metrics:** Some metrics take an extremely long time to compute (in particular FactorVAE). You can change which metrics are computed by adapting `configs/metrics.gin`.
- **Restricting the dataset size:** If you want to quickly test something, and not train on the full dataset, you can do so by changing `main.dataset_size = 0` in the config file to a non-zero value.

## Citing

If you make use of our ideas or this codebase in your research, please consider citing our reports for stage 1 or 2.

Citation information coming soon.