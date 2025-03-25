import os
import subprocess
import tempfile
import cv2
from pathlib import Path
import glob
from horus import util
from horus import project_manager


def make_video_list_file(video_files: list[str]):
    file_name = tempfile.NamedTemporaryFile(delete=False, suffix='.txt').name
    with open(file_name, 'w') as f:
        for video_file in video_files:
            video_path = Path(video_file).resolve()
            f.write(f"file '{video_path}'\n")
    
    return file_name


def run_any_to_av1(input_file_path: str, out_file_path: str):
    command = [
        "ffmpeg",
        "-i", input_file_path,
        "-c:v", "av1_nvenc",
        "-preset", "p1",
        "-tune", "ull",
        "-b:v", "2000k",
        out_file_path
    ]
    try:
        subprocess.run(command, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("Success: any_to_av1")
    except subprocess.CalledProcessError as e:
        print("Failed: any_to_av1()")
        print(e.stderr)


def run_video_contact(input_list: list[str], output_file: str):
    video_files = util.natural_sort(input_list)
    video_list_path = make_video_list_file(video_files)
    command = [
        "ffmpeg",
        "-safe", "0",
        "-c:v", "av1_cuvid",
        "-f", "concat",
        "-i", video_list_path,
        "-c", "copy",
        output_file
    ]

    try:
        subprocess.run(command, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("Success: video_contact")
    except subprocess.CalledProcessError as e:
        print("Failed: video_contact")
        print(e.stderr)


def make_timelaps(input_file: str, output_file: str, max_time_sec: int):
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
        subprocess.run(command, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("Success: make_timelaps")
    except subprocess.CalledProcessError as e:
        print("Failed: make_timelaps")
        print(e.stderr)


def convert_any_to_av1_format(video_files: list[str], out_dir: str):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for video_path in video_files:
        basename = os.path.basename(video_path)
        print("Processing start: ", basename)
        outfilename = basename.replace(os.path.splitext(basename)[1], ".webm")
        outpath = os.path.join(out_dir, outfilename)
        run_any_to_av1(video_path, outpath)


def video_processing_master(raw_video_files: list[str], project_name: str):
    # encode to AV1 codec
    encoded_file_dir = project_manager.make_dir(project_name, "videos")
    convert_any_to_av1_format(raw_video_files, encoded_file_dir)
    util.remove_files(raw_video_files)

    # make merge video
    video_files = util.natural_sort(glob.glob(os.path.join(encoded_file_dir, "*")))
    merge_video_path = project_manager.get_path(project_name, "all_video_merge.webm")
    run_video_contact(video_files, merge_video_path)
    project_manager.edit_project_info_str(
        key="merge_video_name",
        project_name=project_name,
        data="all_video_merge.webm")

    # make timelaps video
    timelaps_video_path = project_manager.get_path(project_name, "timelaps.mp4")
    make_timelaps(merge_video_path, timelaps_video_path, 5 * 60)
    project_manager.edit_project_info_str(
        key="timelaps_video_name",
        project_name=project_name,
        data="timelaps.mp4")

    return merge_video_path, timelaps_video_path
