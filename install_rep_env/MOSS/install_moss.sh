#! /bin/bash
set -e

# git repo
git clone https://github.com/OpenLMLab/MOSS.git || {
    echo "git clone to https://github.com/OpenLMLab/MOSS.git failed!"
    exit 1
}

# enter MOSS
cd MOSS || exit 1

# create moss
conda create --name moss python=3.8 || {
    echo "conda create failed!"
    exit 1
}
conda activate moss
echo "activate moss" || {
    echo "conda create failed!"
    exit 1
}
echo "pip installing requirements.txt ..."
pip install -r requirements.txt
echo "install finished"
