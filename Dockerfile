FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

WORKDIR /workspace

RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime
RUN apt update && apt -y install \
  autoconf \
  automake \
  build-essential \
  cmake \
  git-core \
  libass-dev \
  libfreetype6-dev \
  libgnutls28-dev \
  libmp3lame-dev \
  libsdl2-dev \
  libtool \
  libunistring-dev \
  libva-dev \
  libvdpau-dev \
  libvorbis-dev \
  libx264-dev \
  libx265-dev \
  libxcb-shm0-dev \
  libxcb-xfixes0-dev \
  libxcb1-dev \
  libaom-dev \
  meson \
  nasm \
  ninja-build \
  pkg-config \
  texinfo \
  wget \
  yasm \
  zlib1g-dev \
  git \
  vim \
  curl \
  zip \
  cmake \
  python3 \
  python3-dev \
  python3-pip \
  libgl1-mesa-glx \
  openjdk-11-jdk \
  libvtk7-dev \
  libgtk2.0-dev \
  libgtk-3-dev


COPY ./nv-codec-headers /workspace/nv-codec-headers
RUN cd nv-codec-headers && \
    make install


COPY ./ffmpeg /workspace/ffmpeg
RUN  cd ffmpeg && \
  ./configure \
    --enable-shared \
    --enable-gpl \
    --enable-libx264 \
    --enable-libx265 \
    --enable-libaom \
    --enable-nvdec \
    --enable-nvenc \
    --enable-nonfree && \
  make -j20 && \
  make install && \
  echo "/usr/local/lib" > /etc/ld.so.conf.d/ffmpeg.conf && \
  ldconfig


COPY ./opencv /workspace/opencv
COPY ./opencv_contrib /workspace/opencv_contrib
RUN pip install -U pip && pip install numpy==1.26.4
RUN cd opencv && \
    mkdir build && \
    cd build && \
    cmake -D CMAKE_BUILD_TYPE=Release \
          -D OPENCV_EXTRA_MODULES_PATH=/workspace/opencv_contrib/modules \
          -D WITH_FFMPEG=ON \
          -D OPENCV_FFMPEG_USE_FIND_PACKAGE=OFF \
          -D BUILD_TESTS=OFF \
          -D BUILD_PERF_TESTS=OFF \
          -D BUILD_opencv_calib3d=OFF \
          -D BUILD_opencv_dnn=OFF \
          -D BUILD_opencv_features2d=OFF \
          -D BUILD_opencv_flann=OFF \
          -D BUILD_opencv_gapi=OFF \
          -D BUILD_opencv_ml=OFF \
          -D BUILD_opencv_objdetect=OFF \
          -D BUILD_opencv_photo=OFF \
          -D BUILD_opencv_stitching=OFF \
          -D BUILD_opencv_video=OFF \
          -D BUILD_opencv_python2=OFF \
          -D BUILD_opencv_python3=ON \
          -D PYTHON_VERSION=310 \
          -D PYTHON3_EXECUTABLE=/usr/bin/python3 \
          -D PYTHON3_INCLUDE_DIR=/usr/include/python3.10 \
          -D PYTHON3_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.10.so \
          -D PYTHON3_NUMPY_INCLUDE_DIRS=/usr/local/lib/python3.10/dist-packages/numpy/core/include \
          -D OPENCV_PYTHON3_INSTALL_PATH=/usr/local/lib/python3.10/dist-packages \
          .. && \
    make -j20 && \
    make install && \
    ldconfig


RUN pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
RUN pip install \
    matplotlib==3.7 \
    setuptools==75.6.0 \
    tikzplotlib \
    jpeg4py \
    lmdb \
    pandas \
    scipy \
    loguru \
    flake8 \
    hydra-core \
    iopath \
    psutil \
    py-cpuinfo \
    seaborn \
    ultralytics-thop \
    gradio==4.44.0 \
    ffmpeg-python==0.2.0 \
    gdown==5.2 \
    lapx==0.5.10
RUN pip install --no-deps ultralytics==8.2.63


RUN git clone -b master --single-branch --depth=1 https://github.com/moriyalab/samurai.git
RUN cd /workspace/samurai/sam2/checkpoints && \
    ./download_ckpts.sh && \
    cd ..
