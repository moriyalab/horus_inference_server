FROM ghcr.io/moriyalab/docker-ffmpeg:latest

WORKDIR /workspace
RUN apt update && apt install -y git vim curl python3 python3-pip libgl1-mesa-glx
RUN pip install -U pip && \
  pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 && \
  pip install \
  matplotlib==3.7 \
  setuptools==75.6.0 \
  tikzplotlib \
  jpeg4py \
  opencv-python \
  lmdb \
  pandas \
  scipy \
  loguru \
  flake8 \
  hydra-core \
  iopath \
  ultralytics==8.2.63 \
  gradio==4.44.0 \
  ffmpeg-python==0.2.0 \
  gdown==5.2 \
  lapx==0.5.10

RUN git clone -b master --single-branch --depth=1 https://github.com/moriyalab/samurai.git
RUN cd /workspace/samurai/sam2/checkpoints && \
    ./download_ckpts.sh && \
    cd ..
