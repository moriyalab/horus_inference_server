import cv2
from ultralytics import RTDETR
import time

# モデルのロード
# model = RTDETR('/workspace/horus_inference_server/projects/horus_prj-dc56b22ab7/train_result/weights/best.pt')
model = RTDETR('/workspace/horus_inference_server/projects/horus_prj-dc56b22ab7/train_result/weights/best.engine')

# model.export(
#     format="engine",
#     int8=True,
#     data="/workspace/horus_inference_server/projects/horus_prj-dc56b22ab7/dataset_for_yolo.yaml",  
# )
# # 動画ファイルのパス
video_path = '/workspace/horus_inference_server/projects/horus_prj-dc56b22ab7/timelaps.mp4'

# 動画ファイルを開く
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: 動画ファイルを開けませんでした。")
    exit()

# フレームごとに処理
while True:
    ret, frame = cap.read()
    if not ret:
        break  # 動画の最後に到達

    # モデルによる推論
    start = time.perf_counter()
    results = model.predict(frame, conf=0.4, verbose=False)
    end = time.perf_counter()
    print(int((end - start) * 1000))
    annotated_frame = results[0].plot()

    # 結果の表示
    cv2.imshow("RT-DETR Inference", annotated_frame)

    # キー入力があれば終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# リソース解放
cap.release()
cv2.destroyAllWindows()
