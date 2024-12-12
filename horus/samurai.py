import os
import os.path as osp
import numpy as np
import cv2
import torch
import tempfile
import sys
import random
sys.path.append("/workspace/samurai/sam2")
from sam2.build_sam import build_sam2_video_predictor

color = [(255, 0, 0)]


def project_path_to_dataset_dir(project_path: str):
    dataset_path = os.path.join(project_path, "horus_dataset")
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


def save_yolo_format(bboxes, frame, frame_id, project_path, img_width, img_height):
    save_img_dir, save_label_dir = project_path_to_dataset_dir(project_path)
    save_label_path = os.path.join(save_label_dir, f"{frame_id:08d}.txt")
    save_image_path = os.path.join(save_img_dir, f"{frame_id:08d}.jpg")

    cv2.imwrite(save_image_path, frame)

    with open(save_label_path, "w") as f:
        for obj_id, bbox in bboxes.items():
            x_min, y_min, width, height = bbox
            x_center = (x_min + width / 2) / img_width
            y_center = (y_min + height / 2) / img_height
            norm_width = width / img_width
            norm_height = height / img_height

            f.write(f"{obj_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}\n")

    f.close()


def samurai_inference(video_folder_path: str, project_path: str,  x: int, y: int, w: int, h: int):
    MODEL_PATH = "/workspace/samurai/sam2/checkpoints/sam2.1_hiera_base_plus.pt"
    MODEL_CONFIG = "configs/samurai/sam2.1_hiera_b+.yaml"

    predictor = build_sam2_video_predictor(MODEL_CONFIG, MODEL_PATH, device="cuda:0")
    bbox_prompt = (x, y, x + w, y + h)

    frames = sorted([osp.join(video_folder_path, f) for f in os.listdir(video_folder_path) if f.endswith(".jpg")])
    loaded_frames = [cv2.imread(frame_path) for frame_path in frames]
    height, width = loaded_frames[0].shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    result_video_file_name = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    out = cv2.VideoWriter(result_video_file_name, fourcc, 30, (width, height))

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        state = predictor.init_state(video_folder_path, offload_video_to_cpu=True, offload_state_to_cpu=True, async_loading_frames=True)
        _, _, masks = predictor.add_new_points_or_box(state, box=bbox_prompt, frame_idx=0, obj_id=0)

        for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
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
                frame_id=frame_idx,
                frame=loaded_frames[frame_idx],
                project_path=project_path,
                img_width=width,
                img_height=height
                )

            img = loaded_frames[frame_idx]
            for obj_id, mask in mask_to_vis.items():
                mask_img = np.zeros((height, width, 3), np.uint8)
                mask_img[mask] = color[(obj_id + 1) % len(color)]
                img = cv2.addWeighted(img, 1, mask_img, 0.2, 0)

            for obj_id, bbox in bbox_to_vis.items():
                cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), color[obj_id % len(color)], 2)

            out.write(img)

        out.release()

    return result_video_file_name
