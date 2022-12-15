# start hadoop master container
docker rm -f hadoop-master &> /dev/null
echo "start hadoop-master container..."
docker run -itd \
                --network ray \
                --ip 192.168.1.101 \
                -p 8265:8265 \
                --cap-add=NET_ADMIN \
                --name hadoop-master \
                -v /localdisk/fanli/project/applications.ai.appliedml.workflow.analyticswithpython/:/mnt/code \
                -v /localdisk/fanli/data/recsys2021/:/mnt/data \
                -v /localdisk/fanli/tmp/:/mnt \
                --shm-size=100gb  \
                fanli/hadoop:1.0 &> /dev/null
sleep 5
docker exec -d hadoop-master /bin/bash -c "ip link del dev eth1"
docker exec -d hadoop-master /bin/bash -c "ray start --node-ip-address=192.168.1.101 --head --dashboard-host='0.0.0.0' --dashboard-port=8265"
# get into hadoop master container
docker exec -it hadoop-master bash

