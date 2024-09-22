import gradio as gr
import cv2
from ultralytics import RTDETR
import pandas as pd
import tempfile
import io
import glob


def inference_image(image, mlmodel_name: str, confidence: float):
    model = RTDETR(mlmodel_name)
    results = model.predict(image, conf=confidence / 100)
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


def infer_video(videos, mlmodel_name: str, confidence: float):
    model = RTDETR(mlmodel_name)
    output_files = []
    boxes_info = []
    for video in videos:
        cap = cv2.VideoCapture(video)
        output_frames = []

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                results = model.predict(frame, conf=confidence / 100)

                annotated_frame = results[0].plot()
                output_frames.append(annotated_frame)
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

        if not output_frames:
            return None

        height, width, _ = output_frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        out = cv2.VideoWriter(output_file, fourcc, 30.0, (width, height))

        try:
            for frame in output_frames:
                out.write(frame)
        finally:
            out.release()

        output_files.append(output_file)

    df = pd.DataFrame(boxes_info, columns=["xmin", "ymin", "xmax", "ymax", "confidence", "label"])
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.getvalue()

    return output_files


with gr.Blocks() as main_ui:
    with gr.Tab("Image Inference"):
        gr.Interface(
            inference_image,
            [
                gr.Image(type="numpy", label="Upload an Image"),
                gr.Dropdown(
                    glob.glob("./ml_model/*"), value="rtdetr-l.pt", label="ML Model", info="Will add more animals later!"
                ),
                gr.Slider(0, 100, value=75, label="Confidence", step=5, info="Choose between 0% and 100%"),
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
                gr.File(label="Upload a Video", file_count="multiple", file_types=["mp4", "mpg"]),
                gr.Dropdown(
                    glob.glob("./ml_model/*"), value="rtdetr-l.pt", label="ML Model", info="Will add more animals later!"
                ),
                gr.Slider(0, 100, value=75, label="Confidence", step=5, info="Choose between 0% and 100%"),
            ],
            [
                gr.File(label="Annotated Video"),
                # gr.Textbox(label="Bounding Boxes CSV"),
            ]
        )

if __name__ == "__main__":
    main_ui.launch(server_name="0.0.0.0")
