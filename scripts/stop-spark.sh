#!/bin/bash

N=$1

if [ $# = 0 ]
then
    echo -e "\nstop spark master..."
    $SPARK_HOME/sbin/stop-master.sh
else 
    echo -e "\nstop spark worker..."
    $SPARK_HOME/sbin/stop-worker.sh
fi

echo -e "\nstop spark history..."

$SPARK_HOME/sbin/stop-history-server.sh

echo -e "\n"
