import os
import shutil
import torch
from ultralytics import YOLO
from PIL import Image
import torchvision.transforms as transforms
import owlready2 as or2

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# folders
input_folder = "path/to/data/test/images"
input_labels = "path/to/data/test/labels"

output_folder = "adversarial_test_data/test/images"
output_labels = "adversarial_test_data/test/labels"

os.makedirs(output_folder, exist_ok=True)
os.makedirs(output_labels, exist_ok=True)

# copy labels so validation can still run
for f in os.listdir(input_labels):
    shutil.copy(os.path.join(input_labels, f), os.path.join(output_labels, f))

# load YOLO
yolo_model = YOLO("runs/detect/train/weights/best.pt")
detector = yolo_model.model.to(device)
detector.eval()

# transforms
transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor()
])

to_pil = transforms.ToPILImage()

EPSILON = 0.01

for filename in os.listdir(input_folder):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    path = os.path.join(input_folder, filename)

    img = Image.open(path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    x.requires_grad_(True)

    # raw YOLO output
    preds = detector(x)

    # depending on Ultralytics version, preds may be tuple/list
    if isinstance(preds, (list, tuple)):
        preds = preds[0]

    # YOLOv8 raw output is usually [B, C, N]
    # first 4 channels = bbox, remaining channels = class scores
    cls_scores = preds[:, 4:, :]

    # adversarial goal: LOWER class confidence
    loss = -cls_scores.mean()

    detector.zero_grad()
    loss.backward()

    # FGSM step
    grad_sign = x.grad.sign()
    x_adv = x + EPSILON * grad_sign
    x_adv = torch.clamp(x_adv, 0, 1)

    adv_img = x_adv.squeeze(0).detach().cpu()
    adv_img = to_pil(adv_img)

    save_path = os.path.join(output_folder, filename)
    adv_img.save(save_path)

print("Adversarial dataset generated.")



onto = or2.get_ontology("mlprov.owx").load()
with onto:
    class description(or2.DataProperty):
        range = [str]


req_spec = onto["requirement_specification"]
test_data = onto["testing_dataset"]
model = onto["model"]
map = onto["performance_metric"]

req1 = req_spec("req1")
req1.description = ["The model shall detect instances of the class 'pie' within images."]
orig_test_data = test_data("original_testing_data")
orig_test_data.description = ["github"]
adversarial_test_data = test_data("adversarial_testing_data")
adversarial_test_data.description = ["github"]
model1 = model("Test_Model")
model1.description = ["runs/detect/train/"]

testing_dataset = "path/to/data/"
adversarial_dataset = "adversarial_test_data/"

metrics1 = yolo_model.val(
    data=f"{testing_dataset}/data.yaml",
    imgsz=640,
    batch=16,
    device=0,
    workers=0
)

map1 = map("original_map="+str(metrics1.box.map))

metrics2 = yolo_model.val(
    data=f"{adversarial_dataset}/data.yaml",
    imgsz=640,
    batch=16,
    device=0,
    workers=0
)

map2 = map("adversarial_map="+str(metrics2.box.map))

onto.save(file="mlprov_adversarial_instances.owl")
