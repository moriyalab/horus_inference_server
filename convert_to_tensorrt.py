from ultralytics import RTDETR

model = RTDETR("./ml_model/demo_model.pt")
model.export(format="engine", half=True)
