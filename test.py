# from ultralytics import RTDETR
# import time
# import cv2

# from horus import util

# # モデルのロード
# model = RTDETR('/workspace/horus_inference_server/projects/horus_prj-dc56b22ab7/train_result/weights/best.engine')
# # video_path = '/workspace/horus_inference_server/projects/horus_prj-dc56b22ab7/all_video_merge.webm'
# video_path = '/workspace/horus_inference_server/projects/horus_prj-dc56b22ab7/timelaps.mp4'

# cap = cv2.VideoCapture(video_path)
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# cap.release()

# start = time.perf_counter()
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter("./out.mp4", fourcc, 30, (width, height))


# # inf_data_all = {}
# for index, results in enumerate(model.predict(video_path, stream=True, verbose=False)):
#     # inf_data_all[index] = {}
#     # for box_data in results.boxes:
#     #     box = box_data.xywh[0]
#     #     x_center = max(0, min(int(box[0]), 65535))
#     #     y_center = max(0, min(int(box[1]), 65535))
#     #     width = max(0, min(int(box[2]), 65535))
#     #     height = max(0, min(int(box[3]), 65535))
#     #     inf_data_all[index][int(box_data.cls)] = {
#     #         "x_center": x_center,
#     #         "y_center": y_center,
#     #         "width": width,
#     #         "height": height
#     #     }

#     annotated_frame = results.plot()
#     cv2.imshow("RT-DETR Inference", annotated_frame)
#     out.write(annotated_frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break


#     print(index)

# out.release()

# # util.write_yaml("./out.yaml", inf_data_all)
# end = time.perf_counter()
# print(end - start)

import os
# import gradio as gr
# import cv2
import glob
# import tempfile
from horus import project_manager

project_data = project_manager.get_projects_db()["2024-10-08"]
train_result_dir = os.path.join(project_data["project_path"], "train_result")

print(glob.glob(os.path.join(train_result_dir, "**/*.engine"))[0])
