from ultralytics import RTDETR
import glob


for file in glob.glob("./ml_model/*.pt"):
    model = RTDETR(file)
    model.export(format="engine")
    print(file)
