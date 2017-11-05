#!/usr/bin/env bash
sudo apt-get update && apt-get --assume-yes upgrade
sudo apt-get --assume-yes install tmux build-essential gcc g++ make binutils htop screen
sudo apt-get --assume-yes install software-properties-common unzip tree
