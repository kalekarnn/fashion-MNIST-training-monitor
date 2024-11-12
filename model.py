import torch.nn as nn

class FashionCNN(nn.Module):
    def __init__(self, filters=[16, 32, 64]):
        super(FashionCNN, self).__init__()
        
        # First convolutional layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, filters[0], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Second convolutional layer
        self.conv2 = nn.Sequential(
            nn.Conv2d(filters[0], filters[1], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Third convolutional layer
        self.conv3 = nn.Sequential(
            nn.Conv2d(filters[1], filters[2], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Fourth convolutional layer
        self.conv4 = nn.Sequential(
            nn.Conv2d(filters[2], filters[2], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Fully connected layer
        self.fc = nn.Linear(filters[2] * 1 * 1, 10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x 