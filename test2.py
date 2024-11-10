from moviepy.video.io.VideoFileClip import VideoFileClip
import glob
import os


# 動画分割処理
def split_video(video_path, output_dir, max_duration=600):
    video = VideoFileClip(video_path)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # 動画の長さを確認して、分割が必要な場合のみ処理
    if video.duration > max_duration:
        start_time = 0
        part_index = 1
        
        while start_time < video.duration:
            end_time = min(start_time + max_duration, video.duration)
            subclip = video.subclip(start_time, end_time)
            output_path = os.path.join(output_dir, f"{video_name}_part{part_index}.mp4")
            subclip.write_videofile(output_path, codec="libx264", audio_codec="aac")
            part_index += 1
            start_time = end_time
        
        video.close()
        # 元の長い動画は削除
        # os.remove(video_path)

# 動画分割を行う関数
def split_videos_in_directory(input_dir, output_dir, max_duration=60):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for video_path in glob.glob(os.path.join(input_dir, "*")):
        split_video(video_path, output_dir, max_duration)

# 動画分割を実行
split_videos_in_directory("./videos", "./videos_split")

# 動画パスを新しい分割されたディレクトリに切り替え
video_file_paths = glob.glob("./videos_split/*")
video_names = [os.path.splitext(os.path.basename(video_path))[0] for video_path in video_file_paths]