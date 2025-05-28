#!/bin/bash
set -e
g++ -std=c++17 -fopenmp -o sample_base_analyze.out sample_base_analyze.cpp
g++ sample_hearts_analyze.cpp -o sample_hearts_analyze.out \
  -std=c++17 -fopenmp \
  -O3 \
  -I/usr/local/include/opencv4 \
  -I/usr/local/include \
  -L/usr/local/lib \
  -lopencv_core \
  -lopencv_imgproc \
  -lopencv_highgui \
  -lopencv_imgcodecs \
  -lopencv_videoio

# ./sample_hearts_analyze.out result.csv all_video_merge.webm 0 ./out.csv
# ./sample_base_analyze.cpp CSV_FILE_PATH OUTPUT_DIR