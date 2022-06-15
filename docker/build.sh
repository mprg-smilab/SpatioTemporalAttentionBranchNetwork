#!/bin/bash


# add exec permission to entrypoint.sh
chmod a+x entrypoint.sh


# build
docker build --tag=cumprg/stabn:1.10.0 --force-rm=true --file=./Dockerfile_1_10_0 .
docker build --tag=cumprg/stabn:1.11.0 --force-rm=true --file=./Dockerfile_1_11_0 .


echo "Build docker; done."
