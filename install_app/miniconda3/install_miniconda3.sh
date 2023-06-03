#!/bin/bash

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

source ~/.bashrc