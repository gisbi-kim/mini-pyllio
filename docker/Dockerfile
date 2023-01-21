FROM ubuntu:jammy
# FROM pypose:llio

ENV TZ="Asia/Seoul"
ENV DEBIAN_FRONTEND=noninteractive

RUN \
    apt-get update; \
    apt-get install -y gcc; \
    apt-get install -y g++; \
    apt-get install -y python-pip; \
    apt-get install -y python3-pip; \
    pip install --upgrade pip; \
    pip install ipdb torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu; \
    apt-get install -y python3-opencv; \ 
    apt-get install -y python3-tk; \
    pip install open3d \
    pip install pyyaml; \
    pip install pypose; \
    pip install pykitti
