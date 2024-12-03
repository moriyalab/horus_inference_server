import gradio as gr
import cv2

from horus import util
from horus import inference
from horus import video_processing

# def update_input_video(videoPath: str):
#  frame = util.get_image_from_video(videoPath, 0)
#  return gr.Image(value=frame)


with gr.Blocks() as main_ui:
    with gr.Tab("Upload Video to Database"):
        with gr.Row():
            with gr.Column():
                input_videos = gr.File(label="Upload Video", file_count="multiple", file_types=[".mp4", ".mov", ".mpg"])
                input_project_name = gr.Text(label="Project Name")
                
                upload_button = gr.Button("Start Upload")
            with gr.Column():
                output_status = gr.Text(label="Status")


        # input_videos.change(
        #     update_input_video,
        #     inputs=input_videos,
        #     outputs=[output_preview_video]
        # )

        upload_button.click(
            video_processing.video_processing_ui,
            inputs=[input_videos, input_project_name],
            outputs=[output_status])


if __name__ == "__main__":
    main_ui.queue().launch(server_name="0.0.0.0", server_port=7861)
