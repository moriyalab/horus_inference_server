import cv2
import os
from natsort import natsorted  # 連番の順番を正確にソートするためのライブラリ
import glob

# タイムラプス動画を作成する関数
def create_timelapse(input_dir, output_file, frame_rate):
    # 入力ディレクトリから動画ファイルを取得
    video_files = glob.glob(os.path.join(input_dir, "*.mp4"))  # mp4形式のファイルを取得
    video_files = natsorted(video_files)  # 自然順（連番順）にソート
    
    if not video_files:
        print("動画ファイルが見つかりません")
        return

    # 最初の動画ファイルからフレームサイズを取得
    first_video = cv2.VideoCapture(video_files[0])
    if not first_video.isOpened():
        print("最初の動画ファイルを開けません")
        return

    width = int(first_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(first_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    first_video.release()

    # 出力動画の設定
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 出力フォーマット
    out = cv2.VideoWriter(output_file, fourcc, frame_rate, (width, height))

    # 各動画を読み込み、フレームを結合
    for file in video_files:
        cap = cv2.VideoCapture(file)
        if not cap.isOpened():
            print(f"動画ファイル {file} を開けません")
            continue
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        
        cap.release()
        print(f"{file} を結合しました")

    out.release()
    print(f"タイムラプス動画が作成されました: {output_file}")


# 実行例
input_directory = "path_to_your_directory"  # 動画ファイルが保存されているディレクトリ
output_filename = "timelapse.mp4"          # 出力ファイル名
frame_rate = 30                            # タイムラプスのフレームレート

create_timelapse(input_directory, output_filename, frame_rate)
