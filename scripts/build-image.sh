#!/bin/bash

prox1="$1"
prox2="$2"

echo ""
echo -e "\nbuild docker hadoop image\n"
docker build -t fanli/hadoop:1.0 \
            --build-arg https_proxy=${prox1} \
            --build-arg http_proxy=${prox2} .
echo ""

