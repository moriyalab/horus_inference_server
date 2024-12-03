# import cv2
import os
import glob
import re
import subprocess
import tempfile
from pathlib import Path


def run_ffmpeg_concat(input_list, output_file):
    """
    ffmpegコマンドを使用して、複数の動画を結合する関数。
    
    Parameters:
        input_list (str): 結合するファイルリストを記載したテキストファイル名（例: 'list.txt'）。
        output_file (str): 出力する動画ファイル名（例: 'output.mp4'）。
    """
    command = [
        "ffmpeg",
        "-safe", "0",
        "-f", "concat",
        "-i", input_list,
        "-c", "copy",
        output_file
    ]
    
    try:
        result = subprocess.run(command, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("ffmpegコマンドの実行に成功しました。")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("ffmpegコマンドの実行中にエラーが発生しました。")
        print(e.stderr)


def natural_sort(file_list):
    def alphanum_key(key):
        return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', key)]
    return sorted(file_list, key=alphanum_key)

def merge_videos_and_create_timelapse(input_dir):
    video_files = glob.glob(os.path.join(input_dir, "*.mp4"))
    video_files += glob.glob(os.path.join(input_dir, "*.MOV"))
    video_files += glob.glob(os.path.join(input_dir, "*.mpg"))

    video_files = natural_sort(video_files)
    
    # 各動画を読み込んで結合
    file_name = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    f = open(file_name, 'w')

    for video_file in video_files:
        p = Path(video_file)
        video_path = p.resolve()

        f.write(f"file '{video_path}'\n")
        print(f"{video_path} を処理中...")
    
    return file_name

input_directory = "./2024-10-08"  # 動画ファイルが保存されているディレクトリ

video_index_path =  merge_videos_and_create_timelapse(input_directory)
print(video_index_path)
run_ffmpeg_concat(video_index_path,  "/workspace/horus_inference_server/2024-10-08.mp4")
# ffmpeg -i 2024-10-08.mp4 -c:v libvpx-vp9 -b:v 0 vp9.webm