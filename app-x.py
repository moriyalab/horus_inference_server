import gradio as gr
import cv2

from horus import util
from horus import inference



def update_input_video(videoPath: str):
 frame = util.get_image_from_video(videoPath, 0)
 return gr.Image(value=frame)

with gr.Blocks() as main_ui:
    with gr.Tab("Upload Video to Database"):
        with gr.Row():
            with gr.Column():
                input_video = gr.File(label="Upload Video", file_count="single", file_types=[".mp4", ".mov", ".mpg"])

            with gr.Column():
                output_preview_video = gr.Image(type="numpy", label="Preview Video")


        input_video.change(
            update_input_video,
            inputs=input_video,
            outputs=[output_preview_video]
        )

        # submit_button.click(
        #     inference.main_inference,
        #     inputs=[input_video, input_x, input_y, input_w, input_h],
        #     outputs=[output_video])


if __name__ == "__main__":
    main_ui.queue().launch(server_name="0.0.0.0", server_port=7861)
