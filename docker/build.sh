#!/bin/bash


# add exec permission to entrypoint.sh
chmod a+x entrypoint.sh


# build
docker build --tag=cumprg/stabn:1.10.0 --force-rm=true --file=./Dockerfile .


echo "Build docker; done."
