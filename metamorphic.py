import torch
from ultralytics import YOLO
from roboflow import Roboflow
import owlready2 as or2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

onto = or2.get_ontology("mlprov.owx").load()
with onto:
    class description(or2.DataProperty):
        range = [str]

dataset = "enase26_m1v1/"

testing_dataset = "enase26_test/"

req_spec = onto["requirement_specification"]
pretraining_prov = onto["model"]
train_data = onto["training_dataset"]
test_data = onto["testing_dataset"]
lr = onto["hyperparameter"]
epochs = onto["hyperparameter"]
batch_size = onto["hyperparameter"]
image_size = onto["hyperparameter"]
map = onto["performance_metric"]
model = onto["model"]

req1 = req_spec("req1")
req1.description = ["The model shall detect instances of the class 'pie' within images."]
yolov8 = pretraining_prov("YOLOv8")
training_dataset = train_data("Pie_dataset")
training_dataset.description = ["github"]
pie_test = test_data("Pie_testing")
pie_test.description = ["github"]
learning_rate = lr("Learning_Rate=0.01")
num_epochs = epochs("Number_of_epochs=20")
batchSize = batch_size("Batch_size=16")
imageSize = image_size("Image_size=640x640")


# Model 1
model1 = YOLO("yolov8n.pt") 
                     
results = model1.train(
    data=f"{dataset.location}/data.yaml",
    epochs=20,
    imgsz=640,
    batch=16,
    device=0,
    workers=0
)

metrics1 = model1.val(
    data=f"{testing_dataset}/data.yaml",
    imgsz=640,
    batch=16,
    device=0,
    workers=0
)

model_iteration1 = model("Model_Iteration_1")
model_iteration1.description = ["runs/detect/train/"]
map1 = map("mAP_model_1=" + str(metrics1.box.map))



# Model 2
model2 = YOLO("yolov8n.pt")                           
results = model2.train(
    data=f"{dataset.location}/data.yaml",
    epochs=20,
    imgsz=640,
    batch=16,
    device=0,
    workers=0
) 

metrics2 = model2.val(
    data=f"{testing_dataset}/data.yaml",
    imgsz=640,
    batch=16,
    device=0,
    workers=0
)

model_iteration2 = model("Model_Iteration_2")
model_iteration2.description = ["runs/detect/train2/"]
map2 = map("mAP_model_2=" + str(metrics2.box.map))


# Model 3
model3 = YOLO("yolov8n.pt")                           
results = model3.train(
    data=f"{dataset.location}/data.yaml",
    epochs=20,
    imgsz=640,
    batch=16,
    device=0,
    workers=0
) 

metrics3 = model3.val(
    data=f"{testing_dataset}/data.yaml",
    imgsz=640,
    batch=16,
    device=0,
    workers=0
)

model_iteration3 = model("Model_Iteration_3")
model_iteration3.description = ["runs/detect/train3/"]
map3 = map("mAP_model_3=" + str(metrics3.box.map))


# Model 4
model4 = YOLO("yolov8n.pt")                           
results = model4.train(
    data=f"{dataset.location}/data.yaml",
    epochs=20,
    imgsz=640,
    batch=16,
    device=0,
    workers=0
) 

metrics4 = model4.val(
    data=f"{testing_dataset}/data.yaml",
    imgsz=640,
    batch=16,
    device=0,
    workers=0
)

model_iteration4 = model("Model_Iteration_4")
model_iteration4.description = ["runs/detect/train4/"]
map4 = map("mAP_model_4=" + str(metrics4.box.map))


# Model 5
model5 = YOLO("yolov8n.pt")                           
results = model5.train(
    data=f"{dataset.location}/data.yaml",
    epochs=20,
    imgsz=640,
    batch=16,
    device=0,
    workers=0
) 

metrics5 = model5.val(
    data=f"{testing_dataset}/data.yaml",
    imgsz=640,
    batch=16,
    device=0,
    workers=0
)

model_iteration5 = model("Model_Iteration_5")
model_iteration5.description = ["runs/detect/train5/"]
map5 = map("mAP_model_5=" + str(metrics5.box.map))


onto.save(file="mlprov_metamorphic_instances.owl")