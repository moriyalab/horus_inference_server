import os
import subprocess
import tempfile
import cv2
import uuid
from concurrent.futures import ThreadPoolExecutor
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


def run_ffmpeg_jpeg_to_mp4(input_dir: str, output_file: str):
    f_list = os.path.join(input_dir, "iamge_%06d.jpeg")
    command = [
        "ffmpeg",
        "-r", "30",
        "-i", f_list,
        "-c:v", "h264_nvenc",
        output_file
    ]

    try:
        result = subprocess.run(command, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("ffmpegコマンドの実行に成功しました。")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("ffmpegコマンドの実行中にエラーが発生しました。")
        print(e.stderr)


def _ensure_dir(directory: str):
    if not os.path.exists(directory):
        os.makedirs(directory)


def _make(frame_index: int, output_path: str, video_path: str):
    cap_ = cv2.VideoCapture(video_path)
    if not cap_.isOpened():
        print("Error: Failed to open input video.")
        return
    cap_.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame_data = cap_.read()
    if ret:
        cv2.imwrite(output_path, frame_data)
    else:
        print(f"Error: Failed to read frame {frame_index}.")
    cap_.release()


def run_cv_timelaps_mp4(input_file: str, output_file: str, max_time_sec: int):
    workspace_dir = f'/tmp/horus_ws/{str(uuid.uuid1())[0:11].replace("-", "")}'
    _ensure_dir(workspace_dir)

    cap = cv2.VideoCapture(input_file)
    if not cap.isOpened():
        print("Error: Failed to open input video.")
    frame_size = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    video_time_sec = frame_size / fps
    scale = (video_time_sec / max_time_sec)
    cap.release()

    frame_count = 0
    id = 0
    MAX_THREAD_SIZE = 16
    with ThreadPoolExecutor(MAX_THREAD_SIZE) as executor:
        futures = []
        while frame_count < frame_size:
            output_path = os.path.join(workspace_dir, f"iamge_{id:06}.jpeg")
            futures.append(executor.submit(_make, int(frame_count), output_path, input_file))
            frame_count += scale
            id += 1

        # 全スレッドの完了を待機
        for future in futures:
            future.result()

    run_ffmpeg_jpeg_to_mp4(workspace_dir, output_file)


def video_processing_ui(video_files: list[str], project_name: str):
    project_dir = project_manager.make_project(project_name)

    video_files = util.natural_sort(video_files)
    video_list_path = make_video_list_file(video_files)
    MERGE_V_NAME = "all_video_merge.webm"
    merge_video_path = os.path.join(project_dir, MERGE_V_NAME)
    run_ffmpeg_concat_av1(video_list_path, merge_video_path)
    project_manager.edit_project_info("merge_video_name", MERGE_V_NAME, project_dir)

    TIMELAPS_V_NAME = "timelaps.mp4"
    timelaps_video_path = os.path.join(project_dir, TIMELAPS_V_NAME)
    run_cv_timelaps_mp4(merge_video_path, timelaps_video_path, 5 * 60)
    project_manager.edit_project_info("timelaps_video_name", TIMELAPS_V_NAME, project_dir)

    util.remove_files(video_files)

    return video_files
