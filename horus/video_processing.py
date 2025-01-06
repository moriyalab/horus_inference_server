import os
import subprocess
import tempfile
import cv2
from pathlib import Path
from horus import util
from horus import project_manager


def make_video_list_file(video_files: list[str]):
    file_name = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    f = open(file_name, 'w')

    for video_file in video_files:
        p = Path(video_file)
        video_path = p.resolve()

        f.write(f"file '{video_path}'\n")

    return file_name


def run_ffmpeg_concat_av1(input_list: str, output_file: str):
    command = [
        "ffmpeg",
        "-safe", "0",
        "-f", "concat",
        "-i", input_list,
        "-c:v", "av1_nvenc",
        "-preset", "p1",
        "-tune", "ull",
        "-b:v", "200k",
        output_file
    ]

    try:
        result = subprocess.run(command, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("ffmpegコマンドの実行に成功しました。")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("ffmpegコマンドの実行中にエラーが発生しました。")
        print(e.stderr)


def run_ffmpeg_timelaps_h264(input_file: str, output_file: str, max_time_sec: int):
    cap = cv2.VideoCapture(input_file)
    video_time_sec = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    scale = video_time_sec / max_time_sec

    command = [
        "ffmpeg",
        "-c:v", "av1_cuvid",
        "-i", input_file,
        "-r", "30",
        "-c:v", "h264_nvenc",
        "-b:v", "2000k",
        "-preset", "p1",
        "-tune", "ull",
        "-filter:v", f"setpts={(1.0 / scale)}*PTS",
        output_file
    ]

    try:
        result = subprocess.run(command, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("ffmpegコマンドの実行に成功しました。")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("ffmpegコマンドの実行中にエラーが発生しました。")
        print(e.stderr)


def run_ffmpeg_convert_h264(input_file: str, output_file: str):
    command = [
        "ffmpeg",
        "-i", input_file,
        "-c:v", "h264_nvenc",
        "-preset", "p1",
        "-tune", "ull",
        "-b:v", "2000k",
        output_file
    ]

    try:
        result = subprocess.run(command, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("ffmpegコマンドの実行に成功しました。")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("ffmpegコマンドの実行中にエラーが発生しました。")
        print(e.stderr)


def video_processing_ui(video_files: list[str], project_name: str):
    project_dir = project_manager.make_project(project_name)

    video_files = util.natural_sort(video_files)
    video_list_path = make_video_list_file(video_files)
    MERGE_V_NAME = "all_video_merge.webm"
    merge_video_path = os.path.join(project_dir, MERGE_V_NAME)
    run_ffmpeg_concat_av1(video_list_path, merge_video_path)
    project_manager.edit_project_info_str(
        key="merge_video_name",
        project_dir=project_dir,
        data=MERGE_V_NAME
        )

    TIMELAPS_V_NAME = "timelaps.mp4"
    timelaps_video_path = os.path.join(project_dir, TIMELAPS_V_NAME)
    run_ffmpeg_timelaps_h264(merge_video_path, timelaps_video_path, 5 * 60)
    project_manager.edit_project_info_str(
        key="timelaps_video_name",
        project_dir=project_dir,
        data=TIMELAPS_V_NAME
        )
    util.remove_files(video_files)

    return video_files
