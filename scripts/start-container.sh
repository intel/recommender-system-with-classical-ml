#!/bin/bash

# the default node number is 2
# N=${1:-2}
# start hadoop master container
helpFunction()
{
   echo ""
   echo "Usage: $0 -a parameterM -d parameterD -h parameterH -p parameterP -s parameterS"
   echo -e "\t-m cluster mode, 0 stands for single-node and single-containers, 1 stands for single-node and 2-containers"
   echo -e "\t-d path to the data that needed to be mounted into the container"
   echo -e "\t-h path to the hadoop tmp dir that needed to be mounted into the container"
   echo -e "\t-p your http proxy address, you can find out by typing env in linux command line"
   echo -e "\t-s your https proxy address, you can find out by typing env in linux command line"
   exit 1 # Exit script after printing help
}

while getopts "m:d:h:p:s:" opt
do
   case "$opt" in
      m ) parameterM="$OPTARG" ;;
      d ) parameterD="$OPTARG" ;;
      h ) parameterH="$OPTARG" ;;
      p ) parameterP="$OPTARG" ;;
      s ) parameterS="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

# Print helpFunction in case parameters are empty
if [ -z "$parameterM" ] || [ -z "$parameterD" ] || [ -z "$parameterH" ]|| [ -z "$parameterP" ]|| [ -z "$parameterS" ]
then
   echo "Some or all of the parameters are empty";
   helpFunction
fi


docker rm -f hadoop-master &> /dev/null
echo "start hadoop-master container..."
docker run -itd \
                --net=hadoop \
                -p 8088:8088 \
                -p 8888:8888 \
                -p 8080:8080 \
                -p 9870:9870 \
                -p 9864:9864 \
                -p 4040:4040 \
                -p 18081:18081 \
                -p 10086:22 \
                --name hadoop-master \
                --hostname hadoop-master \
                -v $(pwd)/config/:/mnt/config \
                -v $(pwd):/mnt/code \
                -v ${parameterD}:/mnt/data \
                -v ${parameterH}:/mnt \
                -e https_proxy=${parameterS} \
                -e http_proxy=${parameterP} \
                fanli/hadoop:1.0 &> /dev/null


if [[ $parameterM = "1" ]]
then
    #start hadoop slave container
    i=1
    while [ $i -lt $N ]
    do
    docker rm -f hadoop-slave$i &> /dev/null
        echo "start hadoop-slave$i container..."
        docker run -itd \
                        --net=hadoop \
                        --name hadoop-slave$i \
                        --hostname hadoop-slave$i \
                        fanli/hadoop:1.0 &> /dev/null
        i=$(( $i + 1 ))
    done 
fi 

# get into hadoop master container
docker exec -it hadoop-master bash
