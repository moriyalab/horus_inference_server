import gradio as gr
import cv2
from ultralytics import RTDETR
import pandas as pd
import tempfile
import io
import glob
import shutil
import os

mode_list = ["Create Fully Annotated Video", "Create Time-Lapse Video", "Do Not Create Video"]


def upload_mlmodel(filepaths):
    dest_dir = './ml_model'

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    for filepath in filepaths:
        if os.path.isfile(filepath):
            shutil.copy(filepath, dest_dir)
        else:
            return f'{filepath} is not found'

    return "Upload complete. Please restart gradio_web_ui.py"


def inference_image(image, mlmodel_name: str, confidence: float):
    model = RTDETR(mlmodel_name)
    results = model.predict(
        source=image,
        conf=confidence / 100,
        verbose=False
        )
    annotated_frame = results[0].plot()
    results = results[0].cpu()

    boxes_info = []
    for box_data in results.boxes:
        box = box_data.xywh[0]
        xmin = max(0, min(int(box[0] - box[2] / 2), 65535))
        ymin = max(0, min(int(box[1] - box[3] / 2), 65535))
        xmax = max(0, min(int(box[0] + box[2] / 2), 65535))
        ymax = max(0, min(int(box[1] + box[3] / 2), 65535))
        boxes_info.append([xmin, ymin, xmax, ymax, float(box_data.conf), model.names[int(box_data.cls)]])

    df = pd.DataFrame(boxes_info, columns=["xmin", "ymin", "xmax", "ymax", "confidence", "label"])
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue()
    return annotated_frame, csv_data


def infer_video(videos, mlmodel_name: str, confidence: float, mode: float, progress=gr.Progress()):
    global mode_list
    model = RTDETR(mlmodel_name)
    model.info()
    output_files = []
    boxes_info = []
    for video in videos:
        cap = cv2.VideoCapture(video)
        fps = float(cap.get(cv2.CAP_PROP_FPS))

        if mode_list[0] == mode or mode_list[1] == mode:
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
            out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                progress(cap.get(cv2.CAP_PROP_POS_FRAMES) / cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if not ret:
                    break
                results = model.track(
                    source=frame,
                    verbose=False,
                    persist=True,
                    tracker="botsort.yaml",
                    conf=confidence / 100,
                    half=True
                )

                if mode_list[0] == mode:
                    annotated_frame = results[0].plot()
                    out.write(annotated_frame)
                elif mode_list[1] == mode and cap.get(cv2.CAP_PROP_POS_FRAMES) % int(fps) == 0:
                    annotated_frame = results[0].plot()
                    out.write(annotated_frame)

                results = results[0].cpu()
                for box_data in results.boxes:
                    box = box_data.xywh[0]
                    xmin = max(0, min(int(box[0] - box[2] / 2), 65535))
                    ymin = max(0, min(int(box[1] - box[3] / 2), 65535))
                    xmax = max(0, min(int(box[0] + box[2] / 2), 65535))
                    ymax = max(0, min(int(box[1] + box[3] / 2), 65535))
                    boxes_info.append([xmin, ymin, xmax, ymax, float(box_data.conf), model.names[int(box_data.cls)]])

        finally:
            cap.release()

        out.release()
        output_files.append(output_file)

    df = pd.DataFrame(boxes_info, columns=["xmin", "ymin", "xmax", "ymax", "confidence", "label"])
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.getvalue()

    return output_files


with gr.Blocks() as main_ui:
    with gr.Tab("Upload ML Model"):
        gr.Interface(
            upload_mlmodel,
            [
                gr.File(label="Upload a ml model", file_count="multiple", file_types=["pt", "onnx", "engine"])
            ],
            [
                gr.Textbox(label="Result")
            ]
        )
    with gr.Tab("Image Inference"):
        gr.Interface(
            inference_image,
            [
                gr.Image(type="numpy", label="Upload an Image"),
                gr.Dropdown(
                    glob.glob("./ml_model/*"),
                    value="rtdetr-l.pt",
                    label="ML Model",
                    info="Please place the RT-DETR model in the ml_model directory under the root directory of this project! It supports extensions like .pt, .onnx, and .engine!"
                ),
                gr.Slider(
                    minimum=0,
                    maximum=100,
                    value=75,
                    label="Confidence",
                    step=5,
                    info="Choose between 0% and 100%"
                ),
            ],
            [
                gr.Image(type="numpy", label="result image"),
                gr.Textbox(label="Bounding Boxes CSV"),
            ]
        )
    with gr.Tab("Video Inferemce"):
        gr.Interface(
            infer_video,
            [
                gr.File(label="Upload a Video", file_count="multiple", file_types=["mp4", "mpg", "MOV"]),
                gr.Dropdown(
                    glob.glob("./ml_model/*"),
                    value="rtdetr-l.pt",
                    label="ML Model",
                    info="Please place the RT-DETR model in the ml_model directory under the root directory of this project! It supports extensions like .pt, .onnx, and .engine!"
                ),
                gr.Slider(
                    minimum=0,
                    maximum=100,
                    value=75,
                    label="Confidence",
                    step=5,
                    info="Choose between 0% and 100%"
                ),
                gr.Radio(
                    mode_list,
                    label="Video Creation Options",
                    info="Choose the type of video to create: fully annotated, time-lapse, or none."
                ),
            ],
            [
                gr.File(label="Annotated Video"),
            ]
        )

if __name__ == "__main__":
    main_ui.queue().launch(server_name="0.0.0.0")
