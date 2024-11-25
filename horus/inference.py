from horus import samurai
from horus import util


def main_inference(video_path, x, y, w, h):
    folder_path = util.video_to_images(video_path)
    result_video_path = samurai.samurai_inference(folder_path, x, y, w, h)
    return result_video_path
