import os
from ultralytics import RTDETR
from horus import dataset_manager
from horus import project_manager


def train_base_ml_model(project_name: str):
    project_data = project_manager.get_projects_db()[project_name]
    config_path = os.path.join(
        project_data["project_path"],
        project_data["yolo_dataset"]["config_file"]
    )
    model = RTDETR("rtdetr-l.pt")
    model.train(
        data=config_path,
        epochs=2,
        name="train_result",
        project=project_data["project_path"],
        exist_ok=True,
        imgsz=640
    )
    model.export(
        format="engine",
        int8=True,
        data=config_path,
    )


def ui_build_base_ml_model(project_name: str):
    dataset_manager.convert_to_yolo_dataset(project_name)
    train_base_ml_model(project_name)
    return input
