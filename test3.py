import glob
from threading import Thread, Semaphore
import subprocess
import gc
import os
import shutil
import time

# ファイルを処理する関数
def process_image_with_thread(video_paths):
    # print("Processing start: ", video_paths)
    
    command = [
        "python3", "hoge.py", f"{video_paths}"
    ]
    
    result = subprocess.run(command, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error processing {video_paths}: {result.stderr}")
    else:
        print(result.stdout)

    # print("Processing end: ", video_paths)

# ファイルパスのリストを取得
file_paths = glob.glob("./videos_split/*")

# nで分割する関数
def split_list(lst, n):
    # 1つの分割のサイズを計算
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

# 分割数
n = 20
split_lists = split_list(file_paths, n)

# スレッドのリストを定義
threads = []

# ディレクトリのセットアップ
if os.path.exists("./runs"):
    shutil.rmtree("./runs")
os.makedirs("./runs")

start = time.perf_counter()

# スレッドを作成して開始
for i, sublist in enumerate(split_lists):
    thread = Thread(target=process_image_with_thread, args=(sublist,))
    threads.append(thread)
    thread.start()
    print(f"Group {i + 1}: {sublist}")

# 全てのスレッドが完了するのを待機
for thread in threads:
    thread.join()

end = time.perf_counter()
print(end - start)