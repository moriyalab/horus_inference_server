import gradio as gr
import cv2

from horus import util
from horus import inference

video_path_ = ""

value_x = 0
value_y = 0
value_w = 0
value_h = 0


def update_input_video(videoPath: str):
    global video_path_
    video_path_ = videoPath
    cap = cv2.VideoCapture(videoPath)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frame = update_target_frame(video_path_, 0, 0, 0, 0)

    return gr.update(maximum=width), gr.update(maximum=height), gr.update(maximum=width), gr.update(maximum=height), frame


def update_input_x(value):
    global video_path_, value_x, value_y, value_w, value_h
    value_x = value
    return update_target_frame(video_path_, value_x, value_y, value_w, value_h)


def update_input_y(value):
    global video_path_, value_x, value_y, value_w, value_h
    value_y = value
    return update_target_frame(video_path_, value_x, value_y, value_w, value_h)


def update_input_w(value):
    global video_path_, value_x, value_y, value_w, value_h
    value_w = value
    return update_target_frame(video_path_, value_x, value_y, value_w, value_h)


def update_input_h(value):
    global video_path_, value_x, value_y, value_w, value_h
    value_h = value
    return update_target_frame(video_path_, value_x, value_y, value_w, value_h)


def update_target_frame(video_path, x, y, w, h):
    frame = util.get_image_from_video(video_path, 0)
    start_point = (int(x), int(y))
    end_point = (int(x + w), int(y + h))
    color = (255, 0, 0)
    thickness = 5
    cv2.rectangle(frame, start_point, end_point, color, thickness)
    return gr.Image(value=frame)


with gr.Blocks() as main_ui:
    with gr.Tab("Video Inference"):
        with gr.Row():
            with gr.Column():
                input_video = gr.File(label="Upload Video", file_count="single", file_types=[".mp4", ".mov", ".mpg", "webm"])
                output_image = gr.Image(type="numpy", label="result image")
                input_x = gr.Slider(
                    minimum=0,
                    maximum=100,
                    value=0,
                    label="Input X",
                    step=1,
                )
                input_y = gr.Slider(
                    minimum=0,
                    maximum=100,
                    value=0,
                    label="Input Y",
                    step=1,
                )
                input_w = gr.Slider(
                    minimum=0,
                    maximum=100,
                    value=0,
                    label="Input W",
                    step=1,
                )
                input_h = gr.Slider(
                    minimum=0,
                    maximum=100,
                    value=0,
                    label="Input H",
                    step=1,
                )
                submit_button = gr.Button("Start Inference")
            with gr.Column():
                output_video = gr.File()

        input_video.change(
            update_input_video,
            inputs=input_video,
            outputs=[input_x, input_y, input_w, input_h, output_image]
        )

        input_x.change(update_input_x, inputs=input_x, outputs=[output_image])
        input_y.change(update_input_y, inputs=input_y, outputs=[output_image])
        input_w.change(update_input_w, inputs=input_w, outputs=[output_image])
        input_h.change(update_input_h, inputs=input_h, outputs=[output_image])

        submit_button.click(
            inference.main_inference,
            inputs=[input_video, input_x, input_y, input_w, input_h],
            outputs=[output_video])


if __name__ == "__main__":
    main_ui.queue().launch(server_name="0.0.0.0", server_port=7861)
