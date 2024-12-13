import yaml
import os
import glob
import shutil
import random

from horus import project_manager

def get_label_data(project_path: str):
    labels_dir = os.path.join(project_path, "horus_dataset/labels")
    label_path_list = glob.glob(os.path.join(labels_dir, "*"))
    return label_path_list


def get_image_path(project_path: str):
    images_dir = os.path.join(project_path, "horus_dataset/images")
    images_path_list = glob.glob(os.path.join(images_dir, "*"))
    return images_path_list


def get_all_label_data(label_path_list: list[str]):
    all_label_data = {}
    for filepath in label_path_list:
        with open(filepath, "r") as yml:
            label_data = yaml.safe_load(yml)
            all_label_data[label_data["image_file"]] = label_data["annotations"]
    return all_label_data


def get_all_class(all_label_data, images_path_list):
    cls_list = {}
    for filepath in images_path_list:
        basename = os.path.basename(filepath)
        for key in all_label_data[basename].keys():
            cls_list[key] = {}

    cls_list_with_index = {}
    for index, cls in enumerate(list(cls_list)):
        cls_list_with_index[cls] = index

    return cls_list_with_index

def project_path_to_dataset_dir(project_path: str):
    dataset_path = os.path.join(project_path, "dataset_for_yolo")
    image_train_path = os.path.join(dataset_path, "images/train")
    image_val_path = os.path.join(dataset_path, "images/val")
    label_train_path = os.path.join(dataset_path, "labels/train")
    label_val_path = os.path.join(dataset_path, "labels/val")
    os.makedirs(dataset_path, exist_ok=True)
    os.makedirs(image_train_path, exist_ok=True)
    os.makedirs(image_val_path, exist_ok=True)
    os.makedirs(label_train_path, exist_ok=True)
    os.makedirs(label_val_path, exist_ok=True)

    if random.uniform(0, 100) > 20:
        return image_train_path, label_train_path
    else:
        return image_val_path, label_val_path
    
def conv_ext(path: str, new_ext: str):
    basename_without_ext = os.path.splitext(os.path.basename(path))[0]
    return basename_without_ext + "." + new_ext.replace(".", "")


def make_dataset_yaml(project_path: str, cls_list):
    names = {}
    for cls, id in cls_list.items():
        names[id] = cls

    config_data = {
        "path" : os.path.join(project_path, "dataset_for_yolo"),
        "train" : "images/train",
        "val" : "images/val",
        "names" : names
    }

    with open(os.path.join(project_path, "dataset_for_yolo.yaml"), "w") as yaml_file:
        yaml.dump(config_data, yaml_file, default_flow_style=False, allow_unicode=True)


def main():
    database = project_manager.get_projects_db()
    project_path = database["2024-10-08"]["project_path"]

    label_path_list = get_label_data(project_path)
    images_path_list = get_image_path(project_path)

    all_label_data = get_all_label_data(label_path_list)

    cls_list = get_all_class(all_label_data, images_path_list)


    for image_path in images_path_list:
        save_img_dir, save_label_dir = project_path_to_dataset_dir(project_path)
        img_basename = os.path.basename(image_path)
        lable_basename = conv_ext(img_basename, "txt")
        label_data = all_label_data[img_basename]

        shutil.copy(image_path, save_img_dir)

        with open(os.path.join(save_label_dir, lable_basename), "w") as f:
            for cls, annotation in label_data.items():
                cls_id = cls_list[cls]
                x_center = annotation["x_center"]
                y_center = annotation["y_center"]
                norm_width = annotation["norm_width"]
                norm_height = annotation["norm_height"]

                f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}\n")

        f.close()

    make_dataset_yaml(project_path, cls_list)
