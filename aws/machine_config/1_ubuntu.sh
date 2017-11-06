#!/usr/bin/env bash
sudo apt-get update && apt-get --assume-yes upgrade
sudo apt-get --assume-yes install python3-dev python3-setuptools python3-wheel python3-pip
sudo apt-get --assume-yes install tmux build-essential gcc g++ make binutils htop
sudo apt-get --assume-yes install software-properties-common unzip tree

# Install python3 dependencies
pip3 install -r requirements.txt
# conda env create -n fastai python=3.6 -f environment.yml
# wget http://files.fast.ai/data/dogscats.zip
# conda install psutil tensorflow