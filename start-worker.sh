echo "start hadoop-slave1 container..."
docker rm -f hadoop-slave1 &> /dev/null
docker run -itd \
                --network ray \
                --cap-add=NET_ADMIN \
                --name hadoop-slave1 \
                --shm-size=50gb \
                -v /localdisk/fanli/project/applications.ai.appliedml.workflow.analyticswithpython/:/mnt/code \
                -v /localdisk/fanli/tmp/:/mnt \
                fanli/hadoop:1.0 &> /dev/null
sleep 5
docker exec -d hadoop-slave1 /bin/bash -c "ip link del dev eth1"
docker exec -d hadoop-slave1 /bin/bash -c "ray start --address=192.168.1.101:8265"

# get into hadoop slave container
docker exec -it hadoop-slave1 bash

