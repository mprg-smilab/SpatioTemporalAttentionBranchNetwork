#!/bin/bash


# check command line args
if [ $# -lt 1 ]; then
    echo "ERROR: less arguments"
    echo "USAGE: sh run.sh [container name] [mount volume 1] [mount volume 2] ..."
    exit
fi


# select docker image
imagename="cumprg/stabn:1.11.0"


echo "run docker ..."
echo "    image: ${imagename}"
echo "    container name: ${1}"


# check mount point
if [ $# -gt 1 ]; then
    for var in ${@:2}; do
        mounts+="-v ${var} "
        echo "    mount point: ${var}"
    done
else
    mounts=""
    echo "    mount point: N/A"
fi


# run
docker run --gpus all -ti --rm -u $(id -u):$(id -g) \
        --ipc=host --name=${1} ${mounts} \
        ${imagename}
