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


def run_ffmpeg_concat_av1(input_list: str, output_file: str, preset="fast"):
    command = [
        "ffmpeg",
        "-safe", "0",
        "-f", "concat",
        "-i", input_list,
        "-c:v", "av1_nvenc",
        "-preset", preset,
        "-b:v", "500k",
        output_file
    ]

    try:
        result = subprocess.run(command, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("ffmpegコマンドの実行に成功しました。")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("ffmpegコマンドの実行中にエラーが発生しました。")
        print(e.stderr)


def run_ffmpeg_timelaps_av1(input_file: str, output_file: str, max_time_sec: int):
    cap = cv2.VideoCapture(input_file)
    video_time_sec = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    scale = video_time_sec / max_time_sec

    command = [
        "ffmpeg",
        "-i", input_file,
        "-r", "30",
        "-c:v", "h264_nvenc",
        "-b:v", "700k",
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


def video_processing_ui(video_files: list[str], project_name: str):
    project_dir = project_manager.make_project(project_name)

    video_files = util.natural_sort(video_files)
    video_list_path = make_video_list_file(video_files)
    merge_video_path = os.path.join(project_dir, "all_video_merge.webm")
    run_ffmpeg_concat_av1(video_list_path, merge_video_path)
    project_manager.edit_project_info("merge_video_name", "all_video_merge.webm", project_dir)

    timelaps_video_path = os.path.join(project_dir, "timelaps.mp4")
    run_ffmpeg_timelaps_av1(merge_video_path, timelaps_video_path, 15 * 60)
    project_manager.edit_project_info("timelaps_video_name", "timelaps.mp4", project_dir)

    util.remove_files(video_files)

    return video_files
