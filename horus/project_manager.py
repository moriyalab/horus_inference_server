import uuid
from datetime import datetime
import os
import glob
import subprocess
from horus import util


def make_project(project_name: str, project_host_dir="/workspace/horus_inference_server/projects"):
    project_id = str(uuid.uuid1())[0:11].replace("-", "")
    project_dir = os.path.join(project_host_dir, "horus_prj-" + project_id)
    os.makedirs(project_dir, exist_ok=True)

    project_info_file_path = os.path.join(project_dir, "horus.yaml")

    project_data = {}
    project_data["project_name"] = project_name
    project_data["create_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    util.write_yaml(project_info_file_path, project_data)

    print(f"プロジェクトファイルが作成されました: {project_info_file_path}")
    return project_dir


def get_projects_str(project_host_dir="/workspace/horus_inference_server/projects"):
    project_dir_list = glob.glob(os.path.join(project_host_dir, "*"))

    prj_names = []
    for project in project_dir_list:
        prj_config = util.read_yaml(os.path.join(project, "horus.yaml"))
        prj_names.append(prj_config["project_name"])

    return prj_names


def get_projects_db(project_host_dir="/workspace/horus_inference_server/projects"):
    project_dir_list = glob.glob(os.path.join(project_host_dir, "*"))

    project_database = {}
    for project_path in project_dir_list:
        prj_config = util.read_yaml(os.path.join(project_path, "horus.yaml"))
        prj_config["project_path"] = project_path
        project_database[prj_config["project_name"]] = prj_config

    return project_database


def edit_project_info_str(key: str, project_dir: str, data: str):
    project_info_file = os.path.join(project_dir, "horus.yaml")
    project_data = util.read_yaml(project_info_file)
    project_data[key] = data
    util.write_yaml(project_info_file, project_data)


def edit_project_info_bbox(
        project_dir: str, object_name: str,
        bbox_x_min: int, bbox_y_min: int,
        bbox_width: int, bbox_height: int
        ):
    project_info_file = os.path.join(project_dir, "horus.yaml")
    project_data = util.read_yaml(project_info_file)

    if "base_annotation" not in project_data:
        project_data["base_annotation"] = {}

    project_data["base_annotation"][object_name] = {
            "bbox": {
                "x_min": bbox_x_min,
                "y_min": bbox_y_min,
                "width": bbox_width,
                "height": bbox_height,
            }
        }

    util.write_yaml(project_info_file, project_data)


def edit_project_info_dict(key: str, project_dir: str, data: dict):
    project_info_file = os.path.join(project_dir, "horus.yaml")
    project_data = util.read_yaml(project_info_file)
    project_data[key] = data
    util.write_yaml(project_info_file, project_data)


def remove_project(project_name: str):
    database = get_projects_db()
    project_path = database[project_name]["project_path"]

    command = [
        "rm",
        "-rf",
        project_path
    ]
    try:
        result = subprocess.run(command, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(e.stderr)
