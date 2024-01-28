import torch
from torchvision import datasets, transforms
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import random

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320) 
        x = F.relu(self.fc1(x))
        
        # Use dropout only during training
        if self.training:
            x = F.dropout(x, training=self.training)
        
        x = self.fc2(x)

        return F.softmax(x, dim=1)

model = Net()

# Load the saved model state_dict
model_state_dict = torch.load('neuralNet.pth')
model.load_state_dict(model_state_dict)

model.eval()
model.conv2_drop.training = False

transform = transforms.Compose([transforms.ToTensor()])

# Load the MNIST test dataset
test_data = datasets.MNIST(root='data', train=False, download=True, transform=transform)

def browse_images(label):
    # Get indices of samples with the specified label
    label_indices = [i for i, (img, lbl) in enumerate(test_data) if lbl == label]

    if not label_indices:
        print(f"No samples found with the label {label}.")
        return

    # Randomly select an index from the filtered indices
    random_index = random.choice(label_indices)
    input_image, true_label = test_data[random_index]

    # Run the model on the input image
    with torch.no_grad():
        output = model(input_image.unsqueeze(0))
        predicted_label = output.argmax(dim=1).item()

    # Visualize the input image
    plt.imshow(input_image.squeeze().numpy(), cmap='gray')
    plt.title(f'True Label: {true_label},\n Neural Net Prediction: {predicted_label}')
    plt.show()

# Example: Call browse_images with label 5
user_input = input("Choose a didgit [0-9] and see if the neural network can identify it: ")
browse_images(label=int(user_input))
