import cv2
import os
import uuid
import re


def get_image_from_video(video_path: str, frame_id: int):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Failed to open video.")
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ret, frame = cap.read()

    if not ret:
        print("Failed to read frame.")
        return None

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame


def get_temp_dir():
    temp_dir = os.path.join("/tmp", str(uuid.uuid4()).replace("-", "")[0:10])
    return temp_dir


def video_to_images(video_path: str):
    output_dir = get_temp_dir()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        output_path = os.path.join(output_dir, f"{frame_count:08d}.jpg")
        cv2.imwrite(output_path, frame)

        frame_count += 1

    # 終了処理
    cap.release()

    return output_dir


def natural_sort(file_list):
    def alphanum_key(key):
        filename = os.path.basename(key)
        return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', filename)]
    return sorted(file_list, key=alphanum_key)


def remove_files(file_list: list[str]):
    for file in file_list:
        try:
            os.remove(file)
            print(f"Remove: {file}")
        except:
            print(f"Can't Remove: {file}")
