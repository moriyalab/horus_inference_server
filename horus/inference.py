import os
from horus import samurai
from horus import util
from horus import project_manager


def main_inference(input, x, y, w, h):
    db = project_manager.get_projects_db()
    video_path = os.path.join(db[input]["project_path"], db[input]["timelaps_video_name"])
    folder_path = util.video_to_images(video_path)
    result_video_path = samurai.samurai_inference(folder_path, x, y, w, h)
    return result_video_path
