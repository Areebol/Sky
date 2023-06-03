#! /bin/bash

directory="/home/areebol/miniconda3"
echo "remove dictory $directory"

if [[ -d $directory ]]; then
    rm -rfv "$directory"
else
    echo "$directory does not exist."
fi