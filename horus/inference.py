import os
import gradio as gr
import cv2
import glob
import tempfile
import uuid
import csv
from datetime import datetime
from ultralytics import RTDETR
from horus import samurai
from horus import util
from horus import project_manager
from horus import video_processing


def main_inference(input_project, object_name, x, y, w, h, progress=gr.Progress()):
    db = project_manager.get_projects_db()

    project_manager.edit_project_info_bbox(
        db[input_project]["project_path"], object_name,
        x, y, w, h
    )

    video_path = os.path.join(db[input_project]["project_path"], db[input_project]["timelaps_video_name"])
    video_folder_path = util.video_to_images(video_path)
    result_video_path = samurai.samurai_inference(
        video_folder_path,
        object_name,
        db[input_project]["project_path"],
        x, y, w, h
        )
    return result_video_path


def get_ml_weight(project_name: str, filetype="tensorrt"):
    project_data = project_manager.get_projects_db()[project_name]
    train_result_dir = os.path.join(project_data["project_path"], "train_result")
    if filetype == "tensorrt":
        return glob.glob(os.path.join(train_result_dir, "**/*.engine"))[0]
    elif filetype == "onnx":
        return glob.glob(os.path.join(train_result_dir, "**/*.onnx"))[0]
    else:
        return glob.glob(os.path.join(train_result_dir, "**/*.pt"))[0]


def create_inference_timelaps_video(project_name: str):
    project_data = project_manager.get_projects_db()[project_name]
    video_path = os.path.join(project_data["project_path"], project_data["timelaps_video_name"])

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    result_video_file_name = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(result_video_file_name, fourcc, 30, (width, height))


    model = RTDETR(get_ml_weight(project_name))
    for results in model.predict(video_path, stream=True, verbose=False):
        annotated_frame = results.plot()
        out.write(annotated_frame)
    
    out.release()
    output_video_path = os.path.join(project_data["project_path"], "timelaps_predicted.mp4")
    video_processing.run_ffmpeg_convert_h264(result_video_file_name, output_video_path)

    return output_video_path


def mlanalyze_video(project_name: str):
    project_data = project_manager.get_projects_db()[project_name]
    video_path = os.path.join(project_data["project_path"], project_data["merge_video_name"])

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frameNum = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    anayle_ws_dir = os.path.join(project_data["project_path"], "analyze", str(uuid.uuid1())[0:11].replace("-", ""))
    os.makedirs(anayle_ws_dir, exist_ok=True)

    anayle_info = {
        "project_name": project_name,
        "create_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "video_info" : {
            "name" : project_data["merge_video_name"],
            "width": width,
            "height": height,
            "fps": fps,
            "frameNum": frameNum
        }
    }
    yaml_path = os.path.join(anayle_ws_dir, f"info.yaml")
    util.write_yaml(yaml_path, anayle_info)

    csv_path = os.path.join(anayle_ws_dir, f"result.csv")
    model = RTDETR(get_ml_weight(project_name))
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "class_id", "x_center", "y_center", "width", "height"])

        for index, results in enumerate(model.predict(video_path, stream=True, verbose=False)):
            for box_data in results.boxes:
                box = box_data.xywh[0]
                x_center = util.mxm(box[0])
                y_center = util.mxm(box[1])
                width = util.mxm(box[2])
                height = util.mxm(box[3])
                writer.writerow([index, int(box_data.cls), x_center, y_center, width, height])

    return "OK"