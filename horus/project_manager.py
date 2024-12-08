import uuid
from datetime import datetime
import yaml
import os


def make_project(project_name: str, project_host_dir="/workspace/horus_inference_server/projects"):
    project_id = str(uuid.uuid1())[0:11].replace("-", "")
    project_dir = os.path.join(project_host_dir, "horus_prj-" + project_id)
    os.makedirs(project_dir, exist_ok=True)

    project_description_file_path = os.path.join(project_dir, "horus.yaml")

    project_data = {}
    project_data["project_name"] = project_name
    project_data["create_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(project_description_file_path, 'w') as f:
        yaml.dump(project_data, f, default_flow_style=False, allow_unicode=True)

    print(f"プロジェクトファイルが作成されました: {project_description_file_path}")
    return project_dir
