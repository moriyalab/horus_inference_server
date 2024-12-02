FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

RUN mkdir ~/{.ffmpeg,.ffmpeg-build,.ffmpeg-src}

RUN apt update && apt install -y \
    autoconf \
    automake \
    build-essential \
    cmake \
    git-core \
    libass-dev \
    libfreetype6-dev \
    libgnutls28-dev \
    libtool \
    libvorbis-dev \
    meson \
    ninja-build \
    pkg-config \
    texinfo \
    wget \
    yasm \
    libssl-dev \
    openssl \
    zlib1g-dev \
    nasm \
    libx264-dev \
    libx265-dev libnuma-dev \
    libvpx-dev \
    libmp3lame-dev \
    libopus-dev \
    libaom-dev \
    libfdk-aac-dev \
    git vim curl \
    libavutil-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libffi-dev \
    xz-utils \
    python3-pip

# INSTALL ffmpeg

RUN cd ~/.ffmpeg-src && \
    git clone --depth=1 https://gitlab.com/AOMediaCodec/SVT-AV1.git && \
    cd SVT-AV1/Build && \
    cmake -G "Unix Makefiles" \
      -DCMAKE_INSTALL_PREFIX="$HOME/.ffmpeg-build" \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_DEC=OFF \
      -DBUILD_SHARED_LIBS=OFF .. && \
    make -j4 && \
    make install

RUN wget https://ffmpeg.org/releases/ffmpeg-snapshot.tar.bz2

RUN cd ~/.ffmpeg-src && \
    tar xf ffmpeg-snapshot.tar.bz2 && \
    cd ffmpeg && \
    PKG_CONFIG_PATH="$HOME/.ffmpeg-build/lib/pkgconfig" ./configure \
      --prefix="$HOME/.ffmpeg-build" \
      --pkg-config-flags="--static" \
      --extra-cflags="-I$HOME/.ffmpeg-build/include" \
      --extra-ldflags="-L$HOME/.ffmpeg-build/lib" \
      --extra-libs="-lpthread -lm" \
      --ld="g++" \
      --bindir="$HOME/.ffmpeg/bin" \
      --disable-ffplay \
      --enable-gpl \
      --enable-openssl \
      --enable-libaom \
      --enable-libass \
      --enable-libfdk-aac \
      --enable-libfreetype \
      --enable-libmp3lame \
      --enable-libopus \
      --enable-libsvtav1 \
      --enable-libvorbis \
      --enable-libvpx \
      --enable-libx264 \
      --enable-libx265 \
      --enable-nonfree && \
    make -j8 && \
    make install


RUN pip install --upgrade pip setuptools

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

RUN useradd -u 1000 -m user && \
    chown user:user /workspace
USER user