#! /bin/bash
###
 # @Descripttion: 
 # @version: 1.0
 # @Author: Areebol
 # @Date: 2023-06-06 15:36:58
### 

set -e

if [[ -d 'LLMZoo' ]]; then
    cd LLMZoo
else
    # git repo
    git clone https://github.com/FreedomIntelligence/LLMZoo.git || {
        echo "git clone to https://github.com/FreedomIntelligence/LLMZoo.git failed!"
        exit 1
    }
    # enter LLMZoo
    cd LLMZoo || exit 1
fi

# remove LLMZoo if exits
conda remove --name LLMZoo --all -y || echo "conda path LLMZoo do not exit!"
# create LLMZoo
conda create --name LLMZoo python=3.8 || {
    echo "conda create failed!"
    exit 1
}
echo "pip installing requirements.txt ..."
conda run -n LLMZoo pip install-r requirements.txt
echo "install finished"
