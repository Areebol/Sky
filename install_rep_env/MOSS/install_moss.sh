#! /bin/bash
###
# @Descripttion:
# @version: 1.0
# @Author: Areebol
# @Date: 2023-06-03 21:16:04
 # @LastEditTime: 2023-06-04 00:00:34
###
set -e

if [[ -d 'MOSS' ]]; then
    cd MOSS
else
    # git repo
    git clone https://github.com/OpenLMLab/MOSS.git || {
        echo "git clone to https://github.com/OpenLMLab/MOSS.git failed!"
        exit 1
    }
    # enter MOSS
    cd MOSS || exit 1
fi

# remove moss if exits
conda remove --name moss --all -y || echo "conda path moss do not exit!"
# create moss
conda create --name moss python=3.8 || {
    echo "conda create failed!"
    exit 1
}
echo "pip installing requirements.txt ..."
conda run -n moss pip install-r requirements.txt
echo "install finished"
