import cv2
import os
import glob

def merge_videos_and_create_timelapse(input_dir, output_video, timelapse_speed=2):
    # 動画の連番ファイルを取得（.mp4形式と仮定）
    video_files = sorted(glob.glob(os.path.join(input_dir, "*.mpg")))
    
    if not video_files:
        print("動画ファイルが見つかりません。")
        return

    # 最初の動画を読み込んでプロパティを取得
    first_video = cv2.VideoCapture(video_files[0])
    frame_width = int(first_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(first_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(first_video.get(cv2.CAP_PROP_FPS))
    first_video.release()

    # 出力動画の設定
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4用のコーデック
    output = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

    # 各動画を読み込んで結合
    for video_file in video_files:
        print(f"{video_file} を処理中...")
        cap = cv2.VideoCapture(video_file)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # フレームを指定した速度でタイムラプス化
            if cap.get(cv2.CAP_PROP_POS_FRAMES) % timelapse_speed == 0:
                output.write(frame)
        
        cap.release()

    output.release()
    print(f"タイムラプス動画を保存しました: {output_video}")

# 使用例
input_directory = "./videos"  # 動画ファイルが保存されているディレクトリ
output_file = "output_timelapse.mp4"  # 出力動画のファイル名
timelapse_speed = 300  # 2倍速（2フレームごとに1フレームを抽出）

merge_videos_and_create_timelapse(input_directory, output_file, timelapse_speed)
