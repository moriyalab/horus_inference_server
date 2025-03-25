import os
import os.path as osp
import numpy as np
import cv2
import torch
import tempfile
import sys
import gradio as gr
from horus import util
from horus import video_processing
sys.path.append("/workspace/samurai/sam2")
from sam2.build_sam import build_sam2_video_predictor

color = [(255, 0, 0)]


def project_path_to_dataset_dir(project_path: str, frame_id: int):
    dataset_dir = os.path.join(project_path, "horus_dataset")
    image_dir = os.path.join(dataset_dir, "images")
    label_dir = os.path.join(dataset_dir, "labels")
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    image_name = f"{frame_id:08d}.jpg"
    label_name = f"{frame_id:08d}.yaml"
    image_path = os.path.join(image_dir, image_name)
    label_path = os.path.join(label_dir, label_name)

    return image_path, label_path, image_name, label_name


def save_yolo_format(
        bboxes, object_name: str,
        frame, frame_id: int,
        project_path: str,
        img_width, img_height):
    image_path, label_path, image_name, _ = project_path_to_dataset_dir(project_path, frame_id)

    cv2.imwrite(image_path, frame)

    label_data = util.read_yaml(label_path)

    if "annotations" not in label_data:
        label_data["annotations"] = {}

    label_data["image_file"] = image_name
    for _, bbox in bboxes.items():
        x_min, y_min, width, height = bbox
        x_center = (x_min + width / 2) / img_width
        y_center = (y_min + height / 2) / img_height
        norm_width = width / img_width
        norm_height = height / img_height

        label_data["annotations"][object_name] = {
            "x_center": x_center,
            "y_center": y_center,
            "norm_width": norm_width,
            "norm_height": norm_height,
        }
    util.write_yaml(label_path, label_data)


def samurai_inference(
        video_folder_path: str, object_name: str, project_path: str,
        x: int, y: int, w: int, h: int,
        progress=gr.Progress()):
    MODEL_PATH = "/workspace/samurai/sam2/checkpoints/sam2.1_hiera_base_plus.pt"
    MODEL_CONFIG = "configs/samurai/sam2.1_hiera_b+.yaml"

    predictor = build_sam2_video_predictor(MODEL_CONFIG, MODEL_PATH, device="cuda:0")
    bbox_prompt = (x, y, x + w, y + h)

    frames = sorted([osp.join(video_folder_path, f) for f in os.listdir(video_folder_path) if f.endswith(".jpg")])
    loaded_frames = [cv2.imread(frame_path) for frame_path in frames]
    frame_count = len(loaded_frames)
    image_height, image_width = loaded_frames[0].shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    result_video_file_name = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    out = cv2.VideoWriter(result_video_file_name, fourcc, 30, (image_width, image_height))

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        state = predictor.init_state(video_folder_path, offload_video_to_cpu=True, offload_state_to_cpu=True, async_loading_frames=True)
        _, _, masks = predictor.add_new_points_or_box(state, box=bbox_prompt, frame_idx=0, obj_id=0)

        for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
            progress(frame_idx / frame_count)

            mask_to_vis = {}
            bbox_to_vis = {}

            for obj_id, mask in zip(object_ids, masks):
                mask = mask[0].cpu().numpy()
                mask = mask > 0.0
                non_zero_indices = np.argwhere(mask)
                if len(non_zero_indices) == 0:
                    bbox = [0, 0, 0, 0]
                else:
                    y_min, x_min = non_zero_indices.min(axis=0).tolist()
                    y_max, x_max = non_zero_indices.max(axis=0).tolist()
                    bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
                bbox_to_vis[obj_id] = bbox
                mask_to_vis[obj_id] = mask

            save_yolo_format(
                bboxes=bbox_to_vis,
                object_name=object_name,
                frame_id=frame_idx,
                frame=loaded_frames[frame_idx],
                project_path=project_path,
                img_width=image_width,
                img_height=image_height
                )

            img = loaded_frames[frame_idx]
            for obj_id, mask in mask_to_vis.items():
                mask_img = np.zeros_like(loaded_frames[frame_idx], dtype=np.uint8)
                mask_img[mask.astype(bool)] = color[(obj_id + 1) % len(color)]
                img = cv2.addWeighted(img, 1, mask_img, 0.2, 0)

            for obj_id, bbox in bbox_to_vis.items():
                x_min, y_min, width, height = bbox
                cv2.rectangle(img, (x_min, y_min), (x_min + width, y_min + height), color[obj_id % len(color)], 2)

            out.write(img)

        out.release()

    output_path = os.path.join(project_path, f"samurai_result_{object_name}.mp4")
    video_processing.run_any_to_av1(result_video_file_name, output_path)
    return output_path
