#! /bin/bash

directory="/home/areebol/miniconda3"
echo "remove dictory $directory"

if [[ -d $directory ]]; then
    rm -rfv "$directory"
    echo "$directory remove successfully"
else
    echo "$directory does not exist"
fi