#!/usr/bin/env bash

# Set up training environment.
# Feel free to change these as required:
export AICROWD_EVALUATION_NAME=my_experiment
export AICROWD_DATASET_NAME=mpi3d_toy

# Change these only if you know what you're doing:
# Check if the root is set; if not use the location of this script as root
if [ ! -n "${NDC_ROOT+set}" ]; then
  export NDC_ROOT="$( cd "$(dirname "$0")" ; pwd -P )"
fi

export PYTHONPATH=${PYTHONPATH}:${NDC_ROOT}
export AICROWD_OUTPUT_PATH=${NDC_ROOT}/scratch/shared
export DISENTANGLEMENT_LIB_DATA=${NDC_ROOT}/scratch/dataset
export TORCH_HOME=${NDC_ROOT}/torch_home