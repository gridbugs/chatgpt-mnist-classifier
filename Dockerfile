FROM ubuntu

RUN apt-get update && apt-get upgrade -y && apt-get install -y python3 python3-torch python3-numpy python3-torchvision
COPY x.py .
RUN python3 -u x.py
