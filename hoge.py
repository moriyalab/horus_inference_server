import sys
import time
from threading import Thread, Semaphore
from queue import Queue
from ultralytics import RTDETR
import glob
import os
import shutil
import gc
import time

def process_video(video_path):
    print(f"Processing video: {video_path}")
    model = RTDETR("ml_model/demo_model.engine")
    for _ in model.predict(
                video_path, project="./runs", save=False, save_txt=True,
                save_conf=True, half=True, verbose=True, stream=True
            ):
                pass

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: No video paths provided.")
        sys.exit(1)

    video_paths = sys.argv[1].split(',')

    # 各動画を処理
    for video_path in video_paths:
        process_video(video_path.replace("[", "").replace("]", "").replace("'", "").replace(" ", ""))

    print("All videos processed.")