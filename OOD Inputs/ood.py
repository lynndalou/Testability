import owlready2 as or2

onto = or2.get_ontology("mlprov.owx").load()

req_spec = onto["requirement_specification"]
train_data = onto["training_dataset"]
test_data = onto["testing_dataset"]
model = onto["model"]
iou = onto["performance_metric"]
selected_feature = onto["selected_feature"]
pred = onto["prediction"]


with onto:
    class description(or2.DataProperty):
        range = [str]

req1 = req_spec("req1")
req1.description = ["The model shall detect instances of the 'surfboard' class within images."]


train_dataset_control = train_data("Control_Training_Dataset")
train_dataset_control.description = ["http://images.cocodataset.org/zips/train2017.zip"]
test_dataset = test_data("Control_Testing_Dataset")
test_dataset.description = ["https://github.com/lynndalou/Testability/OOD"]

selected_feature_1 = selected_feature("Beach")
selected_feature_2 = selected_feature("People")
selected_feature_3 = selected_feature("Beach_and_People")

control_model = model("Control_Model")
control_model.description = ["FasterRCNN_COCO2017"]

prediction = pred("Prediction1")
prediction.description = ["surfboard"]
pred_log = []
pred_log.append(prediction)

mean_iou_control = iou("Mean_IoU_of_Control_Model")
mean_iou_control.description = ["0.02"]


onto.save(file="mlprov_ood_instances.owl")