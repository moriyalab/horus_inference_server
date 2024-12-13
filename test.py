# from horus import video_processing

# video_processing.run_ffmpeg_timelaps_h264("/workspace/horus_inference_server/all_video_merge.webm", "/workspace/horus_inference_server/output.mp4", 60)
import os
import cv2


# cap = cv2.VideoCapture("./timelaps.mp4")
# frame_count = 0

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     output_path = os.path.join("./images", f"{frame_count:08d}.jpg")
#     cv2.imwrite(output_path, frame)

#     frame_count += 1

# # 終了処理
# cap.release()

# from ultralytics import RTDETR

# # Load a COCO-pretrained RT-DETR-l model
# model = RTDETR("rtdetr-l.pt")

# # Display model information (optional)
# model.info()

# # Train the model on the COCO8 example dataset for 100 epochs
# results = model.train(data="demo.yml", 
#                       epochs=5, 
#                       name="horus_project",
#                       project="./runs",
#                       exist_ok=True,
#                       imgsz=640)
import shutil
import random

# def yolo2ultralytics():
#     output_yolo_dir = './datasets/horus_datasets'

#     if not os.path.exists(output_yolo_dir):
#         os.makedirs(output_yolo_dir, exist_ok=True)
#         os.makedirs(output_yolo_dir + "/images/train", exist_ok=True)
#         os.makedirs(output_yolo_dir + "/images/val", exist_ok=True)
#         os.makedirs(output_yolo_dir + "/labels/train", exist_ok=True)
#         os.makedirs(output_yolo_dir + "/labels/val", exist_ok=True)

#     files = os.listdir("./yolo_annotations")
#     for file in files:
#         label_file_path = os.path.join("./yolo_annotations", file)
#         image_file_path = os.path.join('./images', file.replace(".txt", ".jpg"))
#         if random.uniform(0, 100) > 20:
#             shutil.copy(label_file_path, output_yolo_dir + "/labels/train")
#             shutil.copy(image_file_path, output_yolo_dir + "/images/train")
#         else:
#             shutil.copy(label_file_path, output_yolo_dir + "/labels/val")
#             shutil.copy(image_file_path, output_yolo_dir + "/images/val")


# yolo2ultralytics()


import yaml

if not os.path.isfile('config.yml'):
    with open('config.yml', mode='w') as f:
        f.close()
with open('config.yml', 'r') as yml:
    config = yaml.safe_load(yml)
    if config == None:
        config = {}
    print(config)