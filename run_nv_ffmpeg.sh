#!/bin/bash

docker run --rm -it \
  --runtime=nvidia \
  -u $(id -u):$(id -g) \
  -v $(pwd):/workspace \
  horus_ffmpeg \
  /bin/bash

# ffmpeg -i 2024-10-08.mp4 -c:v av1_nvenc -c:a copy output_av1.webm

# ffmpeg -i 2024-10-08.mp4 -c:v av1_nvenc -b:v 500k -c:a copy output_av1.webm
# ffmpeg -i 2024-10-08.mp4 -c:v av1_nvenc -b:v 500k -c:a copy output_av1.webm

# ffmpeg -i 010.mpg -vf thumbnail=10 -frames:v 1 output.png
# ffmpeg -i 010.mpg -c:v av1_nvenc -b:v 500k -c:a copy output_av1.webm