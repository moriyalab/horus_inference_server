# WebUI Quick-Start

## 必要なもの
- Docker
- NVIDIA Container Toolkit

## セットアップ

1. Ubuntuのターミナルを開き下記を入力してください。
    ```bash
    cd ~/horus_inference_server
    ./run_dev_env.sh
    ```

2. 下記の画面が表示されます。
    ```bash
    x86
    latest: Pulling from takanotaiga/horus_inference_server
    Digest: sha256:2520dc341eca3538d321f54bfe1306a48902f5e9996c5952bd1230a02541cc97
    Status: Image is up to date for takanotaiga/horus_inference_server:latest
    docker.io/takanotaiga/horus_inference_server:latest

    ==========
    == CUDA ==
    ==========

    CUDA Version 12.4.1

    Container image Copyright (c) 2016-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    This container image and its contents are governed by the NVIDIA Deep Learning Container License.
    By pulling and using the container, you accept the terms and conditions of this license:
    https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license

    A copy of this license is made available in this container at /NGC-DL-CONTAINER-LICENSE for your convenience.

    groups: cannot find name for group ID 1000
    I have no name!@dpc2500015:/workspace/horus_inference_server$ 
    ```

3. この画面が表示されたら下記のコマンドを入力します。

    ```bash
    python3 app-x.py 
    ```

4. URLが表示されるのでブラウザでアクセスします。

    ```bash
    Matplotlib created a temporary config/cache directory at /tmp/matplotlib-71n9k2ik because the default path (/.config/matplotlib) is not a writable directory; it is highly recommended to set the MPLCONFIGDIR environment variable to a writable directory, in particular to speed up the import of Matplotlib and to better support multiprocessing.
    WARNING ⚠️ user config directory '/.config/Ultralytics' is not writeable, defaulting to '/tmp' or CWD.Alternatively you can define a YOLO_CONFIG_DIR environment variable for this path.
    Running on local URL:  http://0.0.0.0:7861

    To create a public link, set `share=True` in `launch()`.
    /usr/local/lib/python3.10/dist-packages/gradio/analytics.py:106: UserWarning: IMPORTANT: You are using gradio version 4.44.0, however version 4.44.1 is available, please upgrade. 
    --------
    warnings.warn(
    ```

5. 分析が完了したら`ctrl+c`で終了します。
