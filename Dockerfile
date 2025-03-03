FROM colmap/colmap:latest
MAINTAINER Xuan-Li

RUN apt-get update && \
    apt-get install -y python3-pip && \
    apt-get install -y wget

COPY . /app
WORKDIR /app
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
RUN pip3 install notebook
