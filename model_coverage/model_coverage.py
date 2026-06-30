import os
import torch
from torch import nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import owlready2 as or2


device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

onto = or2.get_ontology("mlprov.owx").load()

constraint = onto["constraint"]
test_data_ont = onto["testing_dataset"]
model_ont = onto["model"]
metric = onto["performance_metric"]

with onto:
    class description(or2.DataProperty):
        range = [str]


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    

model = NeuralNetwork().to(device)
print(model)
transform = transforms.ToTensor()

train_data = datasets.FashionMNIST(root='./data', train=True,  download=True, transform=transform)
test_data  = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_data,  batch_size=64)

test_dataset = test_data_ont("Testing_Dataset")
test_dataset.description = ["FashionMNIST"]


# --- Loss and optimizer ---
loss_fn   = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

# --- Training loop ---
def train(dataloader, model, loss_fn, optimizer):
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            print(f"  batch {batch:4d}/{len(dataloader)}  loss: {loss.item():.4f}")

# --- Eval loop ---
def test(dataloader, model, loss_fn):
    model.eval()
    correct, total_loss = 0, 0.0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            total_loss += loss_fn(pred, y).item()
            correct    += (pred.argmax(1) == y).sum().item()
    n = len(dataloader.dataset)
    print(f"  Accuracy: {100*correct/n:.1f}%  Avg loss: {total_loss/len(dataloader):.4f}")

# --- Run ---
epochs = 5
for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}/{epochs}")
    train(train_loader, model, loss_fn, optimizer)
    test(test_loader,  model, loss_fn)
torch.save(model.state_dict(), 'model.pth')

trained_model = model_ont("model.pth")

print("Done!")




def calculate_neuron_coverage(model, dataloader, device, threshold=0.0):
    """
    Returns the percentage of neurons that activated at least once
    across the entire test dataset.
    """
    # Track which neurons have ever activated
    activated   = {}  # layer_name -> bool tensor (shape: [num_neurons])
    total_count = {}  # layer_name -> int

    hooks = []
    activation_threshold = constraint("Activation_Threshold=0.0")

    def make_hook(name):
        def hook(module, input, output):
            # output shape: [batch_size, num_neurons]
            # A neuron is "activated" if its output > threshold for any sample
            fired = (output > threshold).any(dim=0).cpu()  # shape: [num_neurons]
            if name not in activated:
                activated[name]   = fired
                total_count[name] = output.shape[1]
            else:
                activated[name] = activated[name] | fired
        return hook

    # Register hooks on ReLU layers only
    for name, module in model.named_modules():
        if isinstance(module, nn.ReLU):
            hooks.append(module.register_forward_hook(make_hook(name)))

    # Pass the full test set through
    model.eval()
    with torch.no_grad():
        for X, _ in dataloader:
            X = X.to(device)
            model(X)

    # Clean up hooks
    for h in hooks:
        h.remove()

    # Aggregate results
    total_neurons   = sum(total_count[n] for n in total_count)
    active_neurons  = sum(activated[n].sum().item() for n in activated)
    coverage        = active_neurons / total_neurons * 100

    print(f"\nNeuron Coverage Results:")
    print(f"{'Layer':<40} {'Active':>8} {'Total':>8} {'Coverage':>10}")
    print("-" * 68)
    for name in activated:
        active = activated[name].sum().item()
        total  = total_count[name]
        print(f"{name:<40} {active:>8} {total:>8} {active/total*100:>9.1f}%")
    print("-" * 68)
    print(f"{'TOTAL':<40} {active_neurons:>8} {total_neurons:>8} {coverage:>9.1f}%")

    return coverage


coverage = calculate_neuron_coverage(model, test_loader, device)
total_coverage = metric("Total_Coverage=" + str(coverage))

onto.save(file="mlprov_model_coverage_instances.owl")