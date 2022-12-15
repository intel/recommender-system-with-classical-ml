#!/bin/bash
N=$1

if [ $# = 0 ]
then
    echo -e "\nstart spark master..."
    source $SPARK_HOME/conf/spark-env.sh 
    $SPARK_HOME/sbin/start-master.sh

    echo -e "\ncreate spark history folder..."
    hdfs dfs -mkdir -p /spark/history

    echo -e "\nstart spark history..."
    $SPARK_HOME/sbin/start-history-server.sh
else 
    echo -e "\nstart spark worker..."
    source $SPARK_HOME/conf/spark-env.sh 
    $SPARK_HOME/sbin/start-worker.sh spark://$MAIN:7077 
fi

echo -e "\n"


