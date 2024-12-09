import os
import gradio as gr
from horus import video_processing
from horus import project_manager


def update_db_select(input):
    db = project_manager.get_projects_db()
    vpath = os.path.join(db[input]["project_path"], db[input]["timelaps_video_name"])
    create_date = db[input]["create_date"]
    project_name = db[input]["project_name"]
    return gr.Video(value=vpath), gr.update(value=create_date), gr.update(value=project_name)


def update_reload_prj_list():
    return gr.update(choices=project_manager.get_projects_str()), update_db_select(project_manager.get_projects_str()[0])[0]


def remove_project_ui(input):
    project_manager.remove_project(input)
    db = project_manager.get_projects_db()
    first_key = list(db.keys())[0]
    vpath = os.path.join(db[first_key]["project_path"], db[first_key]["timelaps_video_name"])
    create_date = db[first_key]["create_date"]
    project_name = db[first_key]["project_name"]
    return gr.update(choices=project_manager.get_projects_str()), gr.Video(value=vpath), gr.update(value=create_date), gr.update(value=project_name)


with gr.Blocks() as main_ui:
    with gr.Tab("Upload Video to Database"):
        with gr.Row():
            with gr.Column():
                input_videos = gr.File(label="Upload Video", file_count="multiple", file_types=[".mp4", ".mov", ".mpg", ".avi"])
                input_project_name = gr.Text(label="Project Name")

                upload_button = gr.Button("Start Upload", variant="primary")
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

                output_project_name = gr.Text(label="Project Name")
                output_create_date = gr.Text(label="Create Date")

                remove_project = gr.Button("Remove Project", variant="stop")

            with gr.Column():
                preview_video = gr.Video(label="Preview Video")

        select_project.change(
            update_db_select,
            inputs=select_project,
            outputs=[preview_video, output_create_date, output_project_name])
        reload_prj_list.click(update_reload_prj_list, inputs=[], outputs=[select_project, preview_video])
        remove_project.click(remove_project_ui, inputs=[select_project], outputs=[select_project, preview_video, output_create_date, output_project_name])


if __name__ == "__main__":
    main_ui.queue().launch(server_name="0.0.0.0", server_port=7861)
