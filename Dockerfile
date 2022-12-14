FROM python:3.8-buster
  
RUN DEBIAN_FRONTEND=noninteractive apt-get -y update && \
    DEBIAN_FRONTEND=noninteractive apt-get -y install --no-install-recommends openssh-server ssh wget vim net-tools git-all && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
 
# install java
RUN wget --no-check-certificate -q https://repo.huaweicloud.com/java/jdk/8u201-b09/jdk-8u201-linux-x64.tar.gz && \
    tar -zxvf jdk-8u201-linux-x64.tar.gz && \
    mv jdk1.8.0_201 /opt/jdk1.8.0_201 && \
    rm jdk-8u201-linux-x64.tar.gz
 
# install hadoop 3.3.3
RUN wget --no-check-certificate -q https://dlcdn.apache.org/hadoop/common/hadoop-3.3.3/hadoop-3.3.3.tar.gz && \
    tar -zxvf hadoop-3.3.3.tar.gz && \
    mv hadoop-3.3.3 /opt/hadoop-3.3.3 && \
    rm hadoop-3.3.3.tar.gz
 
# install spark 3.3.0
RUN wget --no-check-certificate -q https://dlcdn.apache.org/spark/spark-3.3.0/spark-3.3.0-bin-hadoop3.tgz && \
    tar -zxvf spark-3.3.0-bin-hadoop3.tgz && \
    mv spark-3.3.0-bin-hadoop3 /opt/spark-3.3.0-bin-hadoop3 && \
    rm spark-3.3.0-bin-hadoop3.tgz
 
ENV HADOOP_HOME=/opt/hadoop-3.3.3
ENV JAVA_HOME=/opt/jdk1.8.0_201
ENV JRE_HOME=$JAVA_HOME/jre
ENV SPARK_HOME=/opt/spark-3.3.0-bin-hadoop3
ENV PYSPARK_PYTHON=/usr/local/bin/python
ENV PATH=$PATH:$JAVA_HOME/bin:$HADOOP_HOME/bin:$HADOOP_HOME/sbin:$SPARK_HOME/bin
ENV PYTHONPATH=$SPARK_HOME/python:$SPARK_HOME/python/lib/py4j-0.10.9-src.zip:$PYTHONPATH
ENV HADOOP_COMMON_LIB_NATIVE_DIR=$HADOOP_HOME/lib/native
ENV HADOOP_OPTS="$HADOOP_OPTS -Djava.library.path=$HADOOP_HOME/lib/native"
ENV LD_LIBRARY_PATH=$HADOOP_HOME/lib/native
 
RUN ssh-keygen -t rsa -f /root/.ssh/id_rsa -P '' && \
    cat /root/.ssh/id_rsa.pub >> /root/.ssh/authorized_keys && \
    sed -i 's/#   Port 22/Port 12345/' /etc/ssh/ssh_config && \
    sed -i 's/#Port 22/Port 12345/' /etc/ssh/sshd_config

RUN pip install --no-cache-dir pyarrow findspark numpy pandas transformers torch pyrecdp sklearn xgboost
CMD ["sh", "-c", "service ssh start; bash"]
