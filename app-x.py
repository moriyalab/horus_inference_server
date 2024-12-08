import gradio as gr
from horus import video_processing


with gr.Blocks() as main_ui:
    with gr.Tab("Upload Video to Database"):
        with gr.Row():
            with gr.Column():
                input_videos = gr.File(label="Upload Video", file_count="multiple", file_types=[".mp4", ".mov", ".mpg"])
                input_project_name = gr.Text(label="Project Name")

                upload_button = gr.Button("Start Upload")
            with gr.Column():
                output_status = gr.Text(label="Status")

        upload_button.click(
            video_processing.video_processing_ui,
            inputs=[input_videos, input_project_name],
            outputs=[output_status])


if __name__ == "__main__":
    main_ui.queue().launch(server_name="0.0.0.0", server_port=7861)
