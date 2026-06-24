import torch
from ultralytics import YOLO

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = YOLO("yolov8n.pt")

dataset = "path/to/dataset"                                     
                         
                
results = model.train(
    data=f"{dataset.location}/data.yaml",
    epochs=20,
    imgsz=640,
    batch=16,
    device=0,
    workers=0,
    verbose=True,
    augment=False,
    deterministic=True
)               