import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import json
import os
from model import FashionCNN
import random
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import argparse


# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filters', type=str, default='16,32,64')
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=10)
    return parser.parse_args()

def save_image(image_tensor, filepath):
    plt.figure(figsize=(2, 2))
    plt.imshow(image_tensor.squeeze(), cmap='gray')
    plt.axis('off')
    plt.savefig(filepath)
    plt.close()

def train_model(name, filters, batch_size=512, epochs=10, input_optimizer='adam'):
    # Initialize model
    model = FashionCNN(filters=filters).to(device)
    
    # Setup optimizer
    if input_optimizer.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    else:
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    criterion = nn.CrossEntropyLoss()
    
    # Data preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load Fashion MNIST dataset
    train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    losses = []
    accuracies = []
    
    print(f"\nTraining model : {name}")
    print(f"Training model with filters: {filters}")
    print(f"Using {input_optimizer} optimizer and batch size {batch_size}")
    print(f"Training on device: {device}\n")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        # Calculate epoch statistics
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        losses.append(epoch_loss)
        accuracies.append(epoch_acc)
        
        print(f'Epoch [{epoch + 1}/{epochs}] - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')
        
        # Save metrics separately after each epoch
        loss_data = {
            'values': losses,
            'epochs': list(range(1, len(losses) + 1)),
            'config': filters,
            'optimizer': input_optimizer,
            'batch_size': batch_size
        }
        
        accuracy_data = {
            'values': accuracies,
            'epochs': list(range(1, len(accuracies) + 1)),
            'config': filters,
            'optimizer': input_optimizer,
            'batch_size': batch_size
        }
        
        # Save separate files for losses and accuracies
        with open(f'static/losses_{name}.json', 'w') as f:
            json.dump(loss_data, f)
            
        with open(f'static/accuracies_{name}.json', 'w') as f:
            json.dump(accuracy_data, f)

    print(f"\nTraining completed for model: {name}")
    print(f"Final Accuracy: {accuracies[-1]:.2f}%\n")

    # Save the trained model
    if not os.path.exists('models'):
        os.makedirs('models')
    torch.save(model.state_dict(), f'models/{name}.pth')

    return model, losses, accuracies