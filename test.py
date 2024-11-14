from threading import Thread, Semaphore
from queue import Queue
from ultralytics import RTDETR
import glob
import os
import shutil
import gc
import time
import pandas as pd
import io
from collections import defaultdict
import csv


def process_image_with_thread(queue, semaphore):
    model = RTDETR("ml_model/demo_model_fp16.engine")
    while not queue.empty():
        video_path = queue.get()
        with semaphore:
            print("Processing start: ", video_path)
            for _ in model.predict(
                video_path, project="./runs", save=False, save_txt=True,
                save_conf=True, half=False, verbose=False, stream=True, vid_stride=15
            ):
                pass
            print("Processing end: ", video_path)
            gc.collect()
        queue.task_done()
    del model

def run_inference():
    max_threads = 60
    semaphore = Semaphore(max_threads)
    queue = Queue()

    # 全ての動画パスをキューに追加
    for video_path in glob.glob("./videos_split/*"):
        queue.put(video_path)

    threads = []
    for _ in range(max_threads):
        thread = Thread(target=process_image_with_thread, args=(queue, semaphore))
        threads.append(thread)
        thread.start()

    # 全ての動画が処理されるのを待つ
    queue.join()

    # スレッドが完全に終了するのを待つ
    for thread in threads:
        thread.join()

# # ディレクトリのセットアップ
# if os.path.exists("./runs"):
#     shutil.rmtree("./runs")
# os.makedirs("./runs")

# start = time.perf_counter()
# run_inference()
# end = time.perf_counter()
# print(end - start)

result_file_paths = glob.glob("./runs/**/*.txt", recursive=True)
video_file_paths = glob.glob("./videos_split/*")
video_names = [os.path.splitext(os.path.basename(path))[0] for path in video_file_paths]

annotated_results = []
for result_path in result_file_paths:
    file_name_parts = os.path.splitext(os.path.basename(result_path))[0].replace("part", '').split('_')

    if len(file_name_parts) == 3:
        video_name, part_id, index = file_name_parts
        annotated_results.append({
            'path': result_path,
            'video_name': video_name,
            'part_id': int(part_id),
            'index': int(index)
        })
    elif len(file_name_parts) == 2:
        video_name, index = file_name_parts
        annotated_results.append({
            'path': result_path,
            'video_name': video_name,
            'part_id': None,
            'index': int(index)
        })

inference_results = defaultdict(list)

for result in annotated_results:
    result_path = result['path']
    index = result['index']
    video_name = result['video_name']
    part_id = result['part_id']

    with open(result_path, 'r') as file:
        reader = csv.reader(file, delimiter=" ")
        detection_data = {}

        for row in reader:
            if len(row) != 6:
                continue

            class_id, x_center, y_center, width, height, confidence = row
            confidence = float(confidence)

            if class_id not in detection_data or confidence > detection_data[class_id]['confidence']:
                detection_data[class_id] = {
                    'x_center': x_center,
                    'y_center': y_center,
                    'width': width,
                    'height': height,
                    'confidence': confidence
                }

        inference_results[video_name].append({
            'index': index,
            'part_id': part_id,
            'detection_data': detection_data
        })

if os.path.exists("./results"):
    shutil.rmtree("./results")
os.makedirs("./results")

for video_name, video_results in inference_results.items():
    part_sorted_results = defaultdict(list)

    for result in video_results:
        part_sorted_results[result["part_id"]].append(result)

    sorted_part_results = sorted(part_sorted_results.items(), key=lambda x: x[0])
    combined_results = defaultdict(list)

    for _, results in sorted_part_results:
        results.sort(key=lambda x: x['index'])
        for result in results:
            for class_id, detection_info in result['detection_data'].items():
                combined_results[class_id].append([
                    detection_info['x_center'], detection_info['y_center'],
                    detection_info['width'], detection_info['height'], detection_info['confidence']
                ])

    for class_id, detections in combined_results.items():
        df = pd.DataFrame(detections, columns=["x_center", "y_center", "width", "height", "confidence"])
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        output_file_path = f"./results/{video_name}_{class_id}.csv"
        
        with open(output_file_path, "w", encoding="utf-8") as file:
            file.write(csv_buffer.getvalue())