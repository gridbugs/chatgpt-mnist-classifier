import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# manually added these two lines to load the data set
import torchvision
train_dataset = torchvision.datasets.MNIST(root=".", train=True, download=True,transform=torchvision.transforms.ToTensor())

# Load the MNIST dataset
mnist = torch.utils.data.DataLoader(train_dataset)

# Define a simple neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 10)
        
    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
# Create an instance of the network
model = Net()

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the network
for epoch in range(10):
    for data, target in mnist:
        # Forward pass
        output = model(data)
        
        # Compute the loss
        loss = criterion(output, target)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Test the network
with torch.no_grad():
    correct = 0
    total = 0
    for data, target in mnist:
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
    print(f'Accuracy: {correct / total}')

