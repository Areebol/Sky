#! /bin/bash
###
# @Descripttion:
# @version: 1.0
# @Author: Areebol
# @Date: 2023-06-03 21:16:04
 # @LastEditTime: 2023-06-10 10:40:18
###
set -e

if [[ -d 'pandallm' ]]; then
    cd pandallm
else
    # git repo
    git clone https://github.com/dandelionsllm/pandallm.git || {
        echo "git clone to https://github.com/dandelionsllm/pandallm.git failed!"
        exit 1
    }
    # enter MOSS
    cd pandallm || exit 1
fi

# remove panda if exits
conda remove --name panda --all -y || echo "conda path panda do not exit!"
# create panda
conda create --name panda python=3.8 || {
    echo "conda create failed!"
    exit 1
}
echo "pip installing requirements.txt ..."
conda run -n panda pip install -r requirements.txt
echo "install finished"
