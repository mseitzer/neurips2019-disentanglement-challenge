#!/bin/bash

set -e

# Root is where this file is.
export NDC_ROOT="$( cd "$(dirname "$0")" ; pwd -P )"

# Source the training environment (see the env variables defined therein) if we are not evaluating
if [ ! -n "${AICROWD_IS_GRADING+set}" ]; then
  # AICROWD_IS_GRADING is not set, so we're not running on the evaluator and it's safe to
  # source the train_environ.sh
  source ${NDC_ROOT}/train_environ.sh
else
  # We're on the evaluator.
  # Add root to python path, since this would usually be done in train_environ.sh
  export PYTHONPATH=${PYTHONPATH}:${NDC_ROOT}
fi

# If you have other dependencies, this would be a nice place to
# add them to your PYTHONPATH:
#export PYTHONPATH=${PYTHONPATH}:path/to/your/dependency

# Pytorch:
# 
# Note: In case of Pytorch, you will have to export your software runtime via 
#       Anaconda (After installing pytorch), as shown here : 
#		https://github.com/AIcrowd/neurips2019_disentanglement_challenge_starter_kit#how-do-i-specify-my-software-runtime-
# 	as pytorch cannot be installed with just `pip`
#
export PYTHONPATH=${PYTHONPATH}:${NDC_ROOT}/pytorch

if [ ! -n "${FEATURES_DIR+set}" ]; then
  export FEATURES_DIR=./scratch/features
fi

if [ "$#" -eq 2 ]; then
	export CONFIG=$1
	export AICROWD_EVALUATION_NAME=$2
else
	export CONFIG=./configs/submission.gin
  export IS_SUBMISSION=1
fi

python ${NDC_ROOT}/pytorch/train_feature_vae.py --features-dir ${FEATURES_DIR} ${CONFIG} ${AICROWD_EVALUATION_NAME}

# Execute the local evaluation
if [ ! -n "${IS_SUBMISSION+set}" ]; then
	echo "----- LOCAL EVALUATION -----"
	python ${NDC_ROOT}/local_evaluation.py
fi
