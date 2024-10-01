#!/bin/bash

cd ./docker
docker build . -f ./Dockerfile.x86 -t ghcr.io/moriyalab/horus_inference_server:latest
ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
docker run --rm --gpus all --runtime nvidia --shm-size=32G -v $ROOT/datasets:/usr/src/datasets -v $ROOT:/home/root --network host ghcr.io/moriyalab/horus_inference_server:latest /app/utils/copy_poetry_lock.sh
