import owlready2 as or2

onto = or2.get_ontology("mlprov.owx").load()

req_spec = onto["requirement_specification"]
test_data = onto["testing_dataset"]
model = onto["model"]
metric = onto["performance_metric"]
instance = onto["data_instance"]
label = onto["label"]
original_labels = []
transformation = onto["preprocessing_step"]
constraint = onto["constraint"]

with onto:
    class description(or2.DataProperty):
        range = [str]


req1 = req_spec("req1")
req1.description = ["The model shall predict the steering angle for the car."]

udacity = test_data("Udacity_Test_Set")
udacity.description = ["https://www.kaggle.com/datasets/juliuspesonen/udacity-challenge-2"]

seed1 = instance("Seed_image_1")
seed1.description = ["path/to/seed/image"]
seed2 = instance("Seed_image_2")
seed2.description = ["path/to/seed/image"]
seed3 = instance("Seed_image_3")
seed3.description = ["path/to/seed/image"]
seed4 = instance("Seed_image_4")
seed4.description = ["path/to/seed/image"]
seed5 = instance("Seed_image_5")
seed5.description = ["path/to/seed/image"]
# Through 1330 seed images

label1 = label("Label_Image_1")
label1.description = ["angle"]
original_labels.append(label1)
label2 = label("Label_Image_2")
label2.description = ["angle"]
original_labels.append(label2)
label3 = label("Label_Image_3")
label3.description = ["angle"]
original_labels.append(label3)
label4 = label("Label_Image_4")
label4.description = ["angle"]
original_labels.append(label4)
label5 = label("Label_Image_5")
label5.description = ["angle"]
original_labels.append(label5)
# Through 1330 seed labels


averaging_blur_level1 = transformation("Averaging_Blur=3x3")
averaging_blur_level2 = transformation("Averaging_Blur=4x4")
averaging_blur_level3 = transformation("Averaging_Blur=5x5")
averaging_blur_level4 = transformation("Averaging_Blur=6x6")
gaussian_blur_level1 = transformation("Gaussian_Blur=3x3")
gaussian_blur_level2 = transformation("Gaussian_Blur=4x4")
gaussian_blur_level3 = transformation("Gaussian_Blur=5x5")
gaussian_blur_level4 = transformation("Gaussian_Blur=6x6")
median_blur_level1 = transformation("Median_Blur=3x3")
median_blur_level2 = transformation("Median_Blur=4x4")
median_blur_level3 = transformation("Median_Blur=5x5")
median_blur_level4 = transformation("Median_Blur=6x6")
bilateral_blur_level1 = transformation("Bilateral_Blur=3x3")
bilateral_blur_level2 = transformation("Bilateral_Blur=4x4")
bilateral_blur_level3 = transformation("Bilateral_Blur=5x5")
bilateral_blur_level4 = transformation("Bilateral_Blur=6x6")
brightness_level1 = transformation("Brightness=10")
brightness_level2 = transformation("Brightness=20")
brightness_level3 = transformation("Brightness=30")
brightness_level4 = transformation("Brightness=40")
brightness_level5 = transformation("Brightness=50")
brightness_level6 = transformation("Brightness=60")
brightness_level7 = transformation("Brightness=70")
brightness_level8 = transformation("Brightness=80")
brightness_level9 = transformation("Brightness=90")
brightness_level10 = transformation("Brightness=100")
contrast_level1 = transformation("Contrast=1.2")
contrast_level2 = transformation("Contrast=1.4")
contrast_level3 = transformation("Contrast=1.6")
contrast_level4 = transformation("Contrast=1.8")
contrast_level5 = transformation("Contrast=2.0")
contrast_level6 = transformation("Contrast=2.2")
contrast_level7 = transformation("Contrast=2.4")
contrast_level8 = transformation("Contrast=2.6")
contrast_level9 = transformation("Contrast=2.8")
contrast_level10 = transformation("Contrast=3.0")
rotation_level1 = transformation("Rotation=3")
rotation_level2 = transformation("Rotation=6")
rotation_level3 = transformation("Rotation=9")
rotation_level4 = transformation("Rotation=12")
rotation_level5 = transformation("Rotation=15")
rotation_level6 = transformation("Rotation=18")
rotation_level7 = transformation("Rotation=21")
rotation_level8 = transformation("Rotation=24")
rotation_level9 = transformation("Rotation=27")
rotation_level10 = transformation("Rotation=30")
scale_level1 = transformation("Scale=1.5")
scale_level2 = transformation("Scale=2.0")
scale_level3 = transformation("Scale=2.5")
scale_level4 = transformation("Scale=3.0")
scale_level5 = transformation("Scale=3.5")
scale_level6 = transformation("Scale=4.0")
scale_level7 = transformation("Scale=4.5")
scale_level8 = transformation("Scale=5.0")
scale_level9 = transformation("Scale=5.5")
scale_level10 = transformation("Scale=6.0")
shear_level1 = transformation("Horizontal_Shear=0.1")
shear_level2 = transformation("Horizontal_Shear=0.2")
shear_level3 = transformation("Horizontal_Shear=0.3")
shear_level4 = transformation("Horizontal_Shear=0.4")
shear_level5 = transformation("Horizontal_Shear=0.5")
shear_level6 = transformation("Horizontal_Shear=0.6")
shear_level7 = transformation("Horizontal_Shear=0.7")
shear_level8 = transformation("Horizontal_Shear=0.8")
shear_level9 = transformation("Horizontal_Shear=0.9")
shear_level10 = transformation("Horizontal_Shear=1.0")
translation_level1 = transformation("Translation=(10,10)")
translation_level2 = transformation("Translation=(20,20)")
translation_level3 = transformation("Translation=(30,30)")
translation_level4 = transformation("Translation=(40,40)")
translation_level5 = transformation("Translation=(50,50)")
translation_level6 = transformation("Translation=(60,60)")
translation_level7 = transformation("Translation=(70,70)")
translation_level8 = transformation("Translation=(80,80)")
translation_level9 = transformation("Translation=(90,90)")
translation_level10 = transformation("Translation=(100,100)")


transformed_image_1 = instance("Transformed_Image_1")
transformed_image_1.description = ["path/to/image"]
transformed_image_2 = instance("Transformed_Image_2")
transformed_image_2.description = ["path/to/image"]
transformed_image_3 = instance("Transformed_Image_3")
transformed_image_3.description = ["path/to/image"]
transformed_image_4 = instance("Transformed_Image_4")
transformed_image_4.description = ["path/to/image"]
transformed_image_5 = instance("Transformed_Image_5")
transformed_image_5.description = ["path/to/image"]
# For all transformed images

transformed_label_1 = label("Transformed_Label_1")
transformed_label_1.description = ["angle"]
transformed_label_2 = label("Transformed_Label_2")
transformed_label_2.description = ["angle"]
transformed_label_3 = label("Transformed_Label_3")
transformed_label_3.description = ["angle"]
transformed_label_4 = label("Transformed_Label_4")
transformed_label_4.description = ["angle"]
transformed_label_5 = label("Transformed_Label_5")
transformed_label_5.description = ["angle"]
# For all transformed labels

test_suite = test_data("Transformed_Test_Suite")
test_suite.description = ["path/to/data"]


autumn = model("Autumn")
autumn.description = ["https://github.com/udacity/self-drivingcar/tree/master/steering-models/community-models/autumn"]
chauffeur = model("Chauffeur")
chauffeur.description = ["https://github.com/udacity/self-drivingcar/tree/master/steering-models/community-models/chauffeur"]
rambo = model("Rambo")
rambo.description = ["https://github.com/udacity/self-drivingcar/tree/master/steering-models/community-models/rambo"]


behavior_threshold1 = constraint("Threshold=0.1")
behavior_threshold2 = constraint("Threshold=0.2")
behavior_threshold3 = constraint("Threshold=0.3")

# inconsistency(group)(threshold)
inconsistency21 = metric("Inconsistency_group2_threshold1")
inconsistency21.description = ["119"]
inconsistency22 = metric("Inconsistency_group2_threshold2")
inconsistency22.description = ["102"]
inconsistency23 = metric("Inconsistency_group2_threshold3")
inconsistency23.description = ["69"]
inconsistency31 = metric("Inconsistency_group3_threshold1")
inconsistency31.description = ["107"]
inconsistency32 = metric("Inconsistency_group3_threshold2")
inconsistency32.description = ["89"]
inconsistency33 = metric("Inconsistency_group3_threshold3")
inconsistency33.description = ["53"]
inconsistency41 = metric("Inconsistency_group4_threshold1")
inconsistency41.description = ["100"]
inconsistency42 = metric("Inconsistency_group4_threshold2")
inconsistency42.description = ["45"]
inconsistency43 = metric("Inconsistency_group4_threshold3")
inconsistency43.description = ["13"]
inconsistency51 = metric("Inconsistency_group5_threshold1")
inconsistency51.description = ["86"]
inconsistency52 = metric("Inconsistency_group5_threshold2")
inconsistency52.description = ["29"]
inconsistency53 = metric("Inconsistency_group5_threshold3")
inconsistency53.description = ["0"]
inconsistency61 = metric("Inconsistency_group6_threshold1")
inconsistency61.description = ["102"]
inconsistency62 = metric("Inconsistency_group6_threshold2")
inconsistency62.description = ["65"]
inconsistency63 = metric("Inconsistency_group6_threshold3")
inconsistency63.description = ["27"]
inconsistency71 = metric("Inconsistency_group7_threshold1")
inconsistency71.description = ["120"]
inconsistency72 = metric("Inconsistency_group7_threshold2")
inconsistency72.description = ["86"]
inconsistency73 = metric("Inconsistency_group7_threshold3")
inconsistency73.description = ["23"]
inconsistency81 = metric("Inconsistency_group8_threshold1")
inconsistency81.description = ["114"]
inconsistency82 = metric("Inconsistency_group8_threshold2")
inconsistency82.description = ["95"]
inconsistency83 = metric("Inconsistency_group8_threshold3")
inconsistency83.description = ["55"]
inconsistency91 = metric("Inconsistency_group9_threshold1")
inconsistency91.description = ["109"]
inconsistency92 = metric("Inconsistency_group9_threshold2")
inconsistency92.description = ["95"]
inconsistency93 = metric("Inconsistency_group9_threshold3")
inconsistency93.description = ["75"]
inconsistency101 = metric("Inconsistency_group10_threshold1")
inconsistency101.description = ["96"]
inconsistency102 = metric("Inconsistency_group10_threshold2")
inconsistency102.description = ["54"]
inconsistency103 = metric("Inconsistency_group10_threshold3")
inconsistency103.description = ["27"]
inconsistency111 = metric("Inconsistency_group11_threshold1")
inconsistency111.description = ["66"]
inconsistency112 = metric("Inconsistency_group11_threshold2")
inconsistency112.description = ["39"]
inconsistency113 = metric("Inconsistency_group11_threshold3")
inconsistency113.description = ["27"]
inconsistency121 = metric("Inconsistency_group12_threshold1")
inconsistency121.description = ["91"]
inconsistency122 = metric("Inconsistency_group12_threshold2")
inconsistency122.description = ["15"]
inconsistency123 = metric("Inconsistency_group12_threshold3")
inconsistency123.description = ["3"]
inconsistency131 = metric("Inconsistency_group13_threshold1")
inconsistency131.description = ["95"]
inconsistency132 = metric("Inconsistency_group13_threshold2")
inconsistency132.description = ["30"]
inconsistency133 = metric("Inconsistency_group13_threshold3")
inconsistency133.description = ["4"]
inconsistency141 = metric("Inconsistency_group14_threshold1")
inconsistency141.description = ["102"]
inconsistency142 = metric("Inconsistency_group14_threshold2")
inconsistency142.description = ["85"]
inconsistency143 = metric("Inconsistency_group14_threshold3")
inconsistency143.description = ["36"]
inconsistency151 = metric("Inconsistency_group15_threshold1")
inconsistency151.description = ["44"]
inconsistency152 = metric("Inconsistency_group15_threshold2")
inconsistency152.description = ["7"]
inconsistency153 = metric("Inconsistency_group15_threshold3")
inconsistency153.description = ["0"]
inconsistency161 = metric("Inconsistency_group16_threshold1")
inconsistency161.description = ["121"]
inconsistency162 = metric("Inconsistency_group16_threshold2")
inconsistency162.description = ["118"]
inconsistency163 = metric("Inconsistency_group16_threshold3")
inconsistency163.description = ["85"]
inconsistency171 = metric("Inconsistency_group17_threshold1")
inconsistency171.description = ["121"]
inconsistency172 = metric("Inconsistency_group17_threshold2")
inconsistency172.description = ["110"]
inconsistency173 = metric("Inconsistency_group17_threshold3")
inconsistency173.description = ["78"]
inconsistency181 = metric("Inconsistency_group18_threshold1")
inconsistency181.description = ["54"]
inconsistency182 = metric("Inconsistency_group18_threshold2")
inconsistency182.description = ["47"]
inconsistency183 = metric("Inconsistency_group18_threshold3")
inconsistency183.description = ["35"]
inconsistency191 = metric("Inconsistency_group19_threshold1")
inconsistency191.description = ["55"]
inconsistency192 = metric("Inconsistency_group19_threshold2")
inconsistency192.description = ["52"]
inconsistency193 = metric("Inconsistency_group19_threshold3")
inconsistency193.description = ["34"]
inconsistency201 = metric("Inconsistency_group20_threshold1")
inconsistency201.description = ["30"]
inconsistency202 = metric("Inconsistency_group20_threshold2")
inconsistency202.description = ["23"]
inconsistency203 = metric("Inconsistency_group20_threshold3")
inconsistency203.description = ["18"]

onto.save(file="mlprov_combinatorial_instances.owl")