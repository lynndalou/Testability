import owlready2 as or2

onto = or2.get_ontology("mlprov.owx").load()

individual = onto["instance"]
operation = onto["preprocessing_step"]
fitness_func = onto["optimizer"]
criteria = onto["inclusion_criteria"]
model = onto["model"]
threshold = onto["constraint"]
metric = onto["performance_metric"]
test_set = onto["testing_dataset"]

with onto:
    class description(or2.DataProperty):
        range = [str]

crossover = operation("Crossover")
mutation = operation("Mutation")
selection = operation("Selection")


fitness_function = fitness_func("Multi-Objective Genetic Algorithms (MOGA)")
fitness_function.description = ["T. Murata and H. Ishibuchi, “MOGA: Multi-objective genetic algorithms,” Proc. of 1995 IEEE International Conference on Evolutionary Computation, pp. 289-294, Perth, Australia, November 1995"]

termination1 = criteria("Reward")
termination2 = criteria("Probability_of_Functional_Fault")
termination3 = criteria("Number_of_Generations")

initial_population = test_set("Initial_Population")
initial_population.description = ["path/to/data"]
final_population = test_set("Final_Population")
final_population.description = ["path/to/data"]

trained_model = model("cart_pole_model.pt")
cart_pole_threshold = threshold("70")

acc_d0_005 = metric("Accuracy_d0.005=0.63")
acc_d0_01 = metric("Accuracy_d0.01=0.63")
acc_d0_05 = metric("Accuracy_d0.05=0.73")
acc_d0_1 = metric("Accuracy_d0.1=0.92")
acc_d0_5 = metric("Accuracy_d0.5=0.95")
acc_d1 = metric("Accuracy_d1=0.97")
acc_d5 = metric("Accuracy_d5=0.84")
acc_d10 = metric("Accuracy_d10=0.79")
acc_d50 = metric("Accuracy_d50=0.77")
acc_d100 = metric("Accuracy_d100=0.77")

prec_d0_005 = metric("Precision_d0.005=0.39")
prec_d0_01 = metric("Precision_d0.01=0.39")
prec_d0_05 = metric("Precision_d0.05=0.81")
prec_d0_1 = metric("Precision_d0.1=0.92")
prec_d0_5 = metric("Precision_d0.5=0.95")
prec_d1 = metric("Precision_d1=0.97")
prec_d5 = metric("Precision_d5=0.84")
prec_d10 = metric("Precision_d10=0.81")
prec_d50 = metric("Precision_d50=0.78")
prec_d100 = metric("Precision_d100=0.78")

rec_d0_005 = metric("Recall_d0.005=0.63")
rec_d0_01 = metric("Recall_d0.01=0.63")
rec_d0_05 = metric("Recall_d0.05=0.73")
rec_d0_1 = metric("Recall_d0.1=0.92")
rec_d0_5 = metric("Recall_d0.5=0.95")
rec_d1 = metric("Recall_d1=0.97")
rec_d5 = metric("Recall_d5=0.84")
rec_d10 = metric("Recall_d10=0.79")
rec_d50 = metric("Recall_d50=0.77")
rec_d100 = metric("Recall_d100=0.77")

onto.save(file="mlprov_sbt_instances.owl")