from horus import video_processing

video_processing.run_any_to_av1("./ws/test.mp4", "./ws/out-1.webm")
video_processing.run_any_to_av1("./ws/test.mp4", "./ws/out-2.webm")

video_processing.video_processing_ui(["./ws/out-1.webm", "./ws/out-2.webm"], "hoge")