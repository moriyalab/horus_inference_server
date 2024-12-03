import os
import re
import subprocess
import uuid
from datetime import datetime
import yaml
import tempfile
from pathlib import Path


def natural_sort(file_list):
    def alphanum_key(key):
        filename = os.path.basename(key)
        return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', filename)]
    return sorted(file_list, key=alphanum_key)


def make_video_list_file(video_files: list[str]):
    file_name = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    f = open(file_name, 'w')

    for video_file in video_files:
        p = Path(video_file)
        video_path = p.resolve()

        f.write(f"file '{video_path}'\n")
    
    return file_name


def run_ffmpeg_concat_h264(input_list: str, output_file: str, preset="fast"):
    """
    ffmpegコマンドを使用して、複数の動画を結合し、H.264形式でエンコードする関数。
    
    Parameters:
        input_list (str): 結合するファイルリストを記載したテキストファイル名（例: 'list.txt'）。
        output_file (str): 出力する動画ファイル名（例: 'output.mp4'）。
        crf (int): エンコード品質を指定するCRF値(デフォルトは23)。値が低いほど高品質。
        preset (str): エンコード速度と圧縮率を調整するプリセット（デフォルトは'medium'）。
                      'ultrafast'から'slower'まで指定可能。
    """
    command = [
        "ffmpeg",
        "-safe", "0",
        "-f", "concat",
        "-i", input_list,
        "-c:v", "libopenh264",  # H.264エンコード
        # "-preset", preset,  # エンコード速度設定
        output_file
    ]
    
    try:
        result = subprocess.run(command, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("ffmpegコマンドの実行に成功しました。")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("ffmpegコマンドの実行中にエラーが発生しました。")
        print(e.stderr)


def make_project(project_name: str, project_host_dir="/workspace/horus_inference_server/projects"):
    project_id = str(uuid.uuid1())[0:11].replace("-", "")
    project_dir = os.path.join(project_host_dir, project_id)
    os.makedirs(project_dir, exist_ok=True)

    project_description_file_path = os.path.join(project_dir, "horus.yaml")

    project_data = {}
    project_data["project_name"] = project_name
    project_data["create_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(project_description_file_path, 'w') as f:
        yaml.dump(project_data, f, default_flow_style=False, allow_unicode=True)

    print(f"プロジェクトファイルが作成されました: {project_description_file_path}")
    return project_dir


def video_processing_ui(video_files: list[str], project_name: str):
    project_dir = make_project(project_name)

    video_files = natural_sort(video_files)
    video_list_path = make_video_list_file(video_files)
    print(video_list_path)
    run_ffmpeg_concat_h264(video_list_path, os.path.join(project_dir, "data.mp4"))

    return video_files





# def run_ffmpeg_concat(input_list, output_file):
#     """
#     ffmpegコマンドを使用して、複数の動画を結合する関数。
    
#     Parameters:
#         input_list (str): 結合するファイルリストを記載したテキストファイル名（例: 'list.txt'）。
#         output_file (str): 出力する動画ファイル名（例: 'output.mp4'）。
#     """
#     command = [
#         "ffmpeg",
#         "-safe", "0",
#         "-f", "concat",
#         "-i", input_list,
#         "-c", "copy",
#         output_file
#     ]
    
#     try:
#         result = subprocess.run(command, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#         print("ffmpegコマンドの実行に成功しました。")
#         print(result.stdout)
#     except subprocess.CalledProcessError as e:
#         print("ffmpegコマンドの実行中にエラーが発生しました。")
#         print(e.stderr)



