FROM ghcr.io/moriyalab/docker-ffmpeg:amd64-7.0.1-cli-ls144 as buildstage
COPY --from=buildstage /buildout/ /

FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    vim \
    curl \
    wget \
    xz-utils \
    libavutil-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    pkg-config \
    build-essential \
    libffi-dev
RUN pip install --upgrade pip setuptools

RUN wget https://johnvansickle.com/ffmpeg/builds/ffmpeg-git-amd64-static.tar.xz \
      && tar xvf ./ffmpeg-git-amd64-static.tar.xz \
      && cp ./ffmpeg*amd64-static/ffmpeg /usr/local/bin/

RUN git clone -b master --single-branch --depth=1 https://github.com/moriyalab/samurai.git
RUN cd /workspace/samurai/sam2 && \
    pip install -e .

RUN cd /workspace/samurai/sam2 && \
    pip install -e ".[notebooks]" && \
    pip install \
    matplotlib==3.7 \
    tikzplotlib \
    jpeg4py \
    opencv-python \
    lmdb \
    pandas \
    scipy \
    loguru \
    flake8 \
    ultralytics==8.2.63 \
    gradio==4.44.0 \
    ffmpeg-python==0.2.0 \
    gdown==5.2 \
    lapx==0.5.10

RUN cd /workspace/samurai/sam2/checkpoints && \
    ./download_ckpts.sh && \
    cd ..

RUN mkdir -p /workspace/horus_inference_server
