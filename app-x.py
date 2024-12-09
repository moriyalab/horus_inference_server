import os
import gradio as gr
from horus import video_processing
from horus import project_manager

def update_db_select(input):
    db = project_manager.get_projects_db()
    vpath = os.path.join(db[input], "timelaps.mp4")
    return gr.Video(value=vpath)

def update_reload_prj_list():
    return gr.update(choices=project_manager.get_projects_str()), update_db_select(project_manager.get_projects_str()[0])


with gr.Blocks() as main_ui:
    with gr.Tab("Upload Video to Database"):
        with gr.Row():
            with gr.Column():
                input_videos = gr.File(label="Upload Video", file_count="multiple", file_types=[".mp4", ".mov", ".mpg", ".avi"])
                input_project_name = gr.Text(label="Project Name")

                upload_button = gr.Button("Start Upload")
            with gr.Column():
                output_status = gr.Text(label="Status")

        upload_button.click(
            video_processing.video_processing_ui,
            inputs=[input_videos, input_project_name],
            outputs=[output_status])
        
    with gr.Tab("Edit Database"):
        with gr.Row():
            with gr.Column():
                reload_prj_list = gr.Button("Reload Project List")
                select_project = gr.Radio(
                        choices=project_manager.get_projects_str(),
                        label="Projects")
                
            with gr.Column():
                preview_video = gr.Video(label="Preview Video")

        select_project.change(update_db_select, inputs=select_project, outputs=[preview_video])
        reload_prj_list.click(update_reload_prj_list,inputs=[],outputs=[select_project, preview_video])


if __name__ == "__main__":
    main_ui.queue().launch(server_name="0.0.0.0", server_port=7861)
