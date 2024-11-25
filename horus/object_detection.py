from ultralytics import RTDETR
import pandas as pd


def inference_image(image, mlmodel_name: str, confidence: float):
    model = RTDETR(mlmodel_name)
    results = model.predict(
        source=image,
        conf=confidence / 100,
        verbose=False
    )
    annotated_frame = results[0].plot()

    results = results[0].cpu()
    boxes_info = []
    for box_data in results.boxes:
        box = box_data.xywh[0]
        x = int(max(0, min(float(box[0]), 65535)))
        y = int(max(0, min(float(box[1]), 65535)))
        w = int(max(0, min(float(box[2]), 65535)))
        h = int(max(0, min(float(box[3]), 65535)))
        conf = int(float(box_data.conf) * 100)
        boxes_info.append([x, y, w, h, conf, model.names[int(box_data.cls)]])

    df = pd.DataFrame(boxes_info, columns=["x", "y", "w", "h", "confidence", "label"])

    return annotated_frame, df
