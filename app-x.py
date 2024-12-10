import os
import gradio as gr
import cv2
from horus import video_processing
from horus import project_manager
from horus import util
from horus import inference


video_path_ = ""

value_x = 0
value_y = 0
value_w = 0
value_h = 0


def update_db_select(input):
    db = project_manager.get_projects_db()
    vpath = os.path.join(db[input]["project_path"], db[input]["timelaps_video_name"])
    create_date = db[input]["create_date"]
    project_name = db[input]["project_name"]
    return gr.Video(value=vpath), gr.update(value=create_date), gr.update(value=project_name)


def update_reload_prj_list():
    return gr.update(choices=project_manager.get_projects_str()), update_db_select(project_manager.get_projects_str()[0])[0]


def update_reload_prj_list_vinf():
    return gr.update(choices=project_manager.get_projects_str())


def remove_project_ui(input):
    project_manager.remove_project(input)
    db = project_manager.get_projects_db()
    first_key = list(db.keys())[0]
    vpath = os.path.join(db[first_key]["project_path"], db[first_key]["timelaps_video_name"])
    create_date = db[first_key]["create_date"]
    project_name = db[first_key]["project_name"]
    return gr.update(choices=project_manager.get_projects_str()), gr.Video(value=vpath), gr.update(value=create_date), gr.update(value=project_name)


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


def update_input_video(input: str):
    global video_path_
    db = project_manager.get_projects_db()
    video_path_ = os.path.join(db[input]["project_path"], db[input]["timelaps_video_name"])
    cap = cv2.VideoCapture(video_path_)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frame = update_target_frame(video_path_, 0, 0, 0, 0)
    return gr.update(maximum=width), gr.update(maximum=height), gr.update(maximum=width), gr.update(maximum=height), frame



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

    with gr.Tab("Video Inference"):
        with gr.Row():
            with gr.Column():
                reload_prj_list_vinf = gr.Button("Reload Project List")
                select_project_vinf = gr.Radio(
                        choices=project_manager.get_projects_str(),
                        label="Projects")
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

        select_project_vinf.change(
            update_input_video,
            inputs=select_project_vinf,
            outputs=[input_x, input_y, input_w, input_h, output_image]
        )
        reload_prj_list.click(update_reload_prj_list_vinf, inputs=[], outputs=[select_project_vinf])

        input_x.change(update_input_x, inputs=input_x, outputs=[output_image])
        input_y.change(update_input_y, inputs=input_y, outputs=[output_image])
        input_w.change(update_input_w, inputs=input_w, outputs=[output_image])
        input_h.change(update_input_h, inputs=input_h, outputs=[output_image])

        submit_button.click(
            inference.main_inference,
            inputs=[select_project_vinf, input_x, input_y, input_w, input_h],
            outputs=[output_video])

if __name__ == "__main__":
    main_ui.queue().launch(server_name="0.0.0.0", server_port=7861)
