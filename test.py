import cv2
import os
from concurrent.futures import ThreadPoolExecutor

# フォルダ作成関数
def ensure_dir(directory: str):
    if not os.path.exists(directory):
        os.makedirs(directory)

# フレーム処理関数
def make(frame: int, file: str):
    cap_ = cv2.VideoCapture(file)
    if not cap_.isOpened():
        print("Error: Failed to open input video.")
        return
    cap_.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ret, frame_data = cap_.read()
    if ret:
        cv2.imwrite(f'./test/out_{frame}.jpg', frame_data)
    else:
        print(f"Error: Failed to read frame {frame}.")
    cap_.release()

if __name__ == "__main__":
    input_file = './all_video_merge.webm'
    max_time_sec = 5 * 60  # 10分
    max_threads = 32  # 最大スレッド数

    # 出力フォルダ確認
    ensure_dir('./test/')

    # 動画情報取得
    cap = cv2.VideoCapture(input_file)
    if not cap.isOpened():
        print("Error: Failed to open input video.")
        exit()

    frame_size = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    video_time_sec = frame_size / fps
    scale = (video_time_sec / max_time_sec)
    cap.release()

    # スレッドプールを使用してフレーム処理
    frame_count = 0
    with ThreadPoolExecutor(max_threads) as executor:
        futures = []
        while frame_count < frame_size:
            futures.append(executor.submit(make, int(frame_count), input_file))
            frame_count += scale

        # 全スレッドの完了を待機
        for future in futures:
            future.result()

    print("Processing completed.")
