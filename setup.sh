#!/bin/bash
if type "conda" > /dev/null; then
  conda env create -f environment.yml
  eval "$(conda shell.bash hook)"
  conda activate cs7643-project
fi
pip install git+https://github.com/facebookresearch/fastMRI.git