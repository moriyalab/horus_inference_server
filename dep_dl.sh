
rm -rf opencv opencv_contrib nv-codec-headers ffmpeg
git clone -b 4.9.0 --depth=1 https://github.com/opencv/opencv.git
git clone -b 4.9.0 --depth=1 https://github.com/opencv/opencv_contrib.git
git clone --branch n12.2.72.0 --depth 1 https://git.videolan.org/git/ffmpeg/nv-codec-headers.git
git clone -b n7.1 --depth=1 https://git.ffmpeg.org/ffmpeg.git