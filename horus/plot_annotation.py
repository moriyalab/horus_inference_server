from horus import project_manager
import os
import cv2


def draw_label(image, text, position, font_scale=1, thickness=2, color=(255, 0, 0), text_color=(255, 255, 255)):
    w, h = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, thickness=thickness)[0]
    h += 5
    x, y = position
    outside = y >= h
    if x > image.shape[1] - w:
        x = image.shape[1] - w
    p2 = (x + w, y - h if outside else y + h)
    cv2.rectangle(image, (x, y), p2, color, -1, cv2.LINE_AA)
    cv2.putText(
        image,
        text,
        (x, y - 2 if outside else y + h - 1),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        text_color,
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )


def draw_bbox(image, x_min, y_min, width, height, back_color):
    cv2.rectangle(
        image,
        (x_min, y_min),
        (x_min + width, y_min + height),
        back_color,
        lineType=cv2.LINE_AA
    )


def plot_base_annotation(project_name: str):
    colors = [
        ((4, 42, 255), (255, 255, 255)),
        ((11, 219, 235), (0, 0, 0)),
        ((243, 243, 243), (0, 0, 0)),
        ((0, 223, 183), (0, 0, 0)),
        ((17, 31, 104), (255, 255, 255)),
        ((255, 111, 221), (255, 255, 255)),
        ((255, 68, 79), (255, 255, 255)),
        ((204, 237, 0), (255, 255, 255)),
        ((0, 243, 68), (0, 0, 0)),
        ((189, 0, 255), (255, 255, 255)),
        ((0, 180, 255), (0, 0, 0))
        ]

    project_data = project_manager.get_projects_db()[project_name]
    video_path = os.path.join(project_data["project_path"], project_data["timelaps_video_name"])
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cap.release()

    if not ret:
        return None

    if "base_annotation" not in project_data:
        project_data["base_annotation"] = {}

    for index, (object_name, annotation) in enumerate(project_data["base_annotation"].items()):
        bbox = annotation["bbox"]
        x_min = bbox["x_min"]
        y_min = bbox["y_min"]
        width = bbox["width"]
        height = bbox["height"]
        back_color, text_color = colors[index]
        draw_bbox(frame, x_min, y_min, width, height, back_color)
        draw_label(frame, object_name, (x_min, y_min), font_scale=0.8, thickness=2, color=back_color, text_color=text_color)

    return frame
