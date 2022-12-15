#!/bin/bash

N=$1

if [ $# = 0 ]
then
    echo -e "\nformat namenode..."
    $HADOOP_HOME/bin/hdfs namenode -format
else 
    echo -e "\nformat datanode..."
    $HADOOP_HOME/bin/hdfs datanode -format
fi


echo -e "\nstart HDFS..."

$HADOOP_HOME/sbin/start-dfs.sh 

echo -e "\n"
