#!/bin/bash
###
 # @Descripttion: 
 # @version: 1.0
 # @Author: Areebol
 # @Date: 2023-06-03 13:20:46
 # @LastEditTime: 2023-06-03 23:18:47
### 
set -e
# download minicodna.sh
echo "Downloading miniconda.sh..."
if [[ -f "Miniconda3-latest-Linux-x86_64.sh" ]]; then
    echo "Download has already been satisfied."
else 
    wget -c https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    echo "Download complete."
fi

chmod 777 Miniconda3-latest-Linux-x86_64.sh

sh Miniconda3-latest-Linux-x86_64.sh

export PATH=/home/areebol/miniconda3/bin:$PATH

source ~/.bashrc || { echo " ~/.bahsrc do not exist!"; exit 1;}