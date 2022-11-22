#!/bin/bash

echo ""
echo -e "\nbuild docker hadoop image\n"
docker build -t recsys2021:1.0 \
            --build-arg https_proxy=${https_proxy} \
            --build-arg HTTPS_PROXY=${HTTPS_PROXY} \
            --build-arg HTTP_PROXY=${HTTP_PROXY} \
            --build-arg http_proxy=${http_proxy} .
echo ""

