import logging
from flask import Flask, render_template, request, jsonify, send_from_directory
import json
import os
import threading
import train
import torch
from torchvision import datasets, transforms
from model import FashionCNN
import random
import matplotlib.pyplot as plt

app = Flask(__name__)

# Disable Flask's default logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# Disable Flask's default access logs
def secho(text, file=None, nl=None, err=None, color=None, **styles):
    pass

import click
click.echo = secho
click.secho = secho

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_metrics')
def get_metrics():
    metrics = {'losses': {}, 'accuracies': {}}
    
    # Get loss data
    for filename in os.listdir('static'):
        if filename.startswith('losses_') and filename.endswith('.json'):
            try:
                with open(f'static/{filename}', 'r') as f:
                    model_name = filename[7:-5]  # Remove 'losses_' and '.json'
                    metrics['losses'][model_name] = json.load(f)
            except json.JSONDecodeError:
                continue

    # Get accuracy data
    for filename in os.listdir('static'):
        if filename.startswith('accuracies_') and filename.endswith('.json'):
            try:
                with open(f'static/{filename}', 'r') as f:
                    model_name = filename[11:-5]  # Remove 'accuracies_' and '.json'
                    metrics['accuracies'][model_name] = json.load(f)
            except json.JSONDecodeError:
                continue
                
    return jsonify(metrics)

def clear_metrics():
    for filename in os.listdir('static'):
        if any(filename.startswith(prefix) for prefix in ['losses_', 'accuracies_']) and filename.endswith('.json'):
            os.remove(os.path.join('static', filename))

@app.route('/start_training', methods=['POST'])
def start_training():
    config = request.json
    
    # Clear existing metrics files
    clear_metrics()
    
    model1_config = config["model1"]
    model2_config = config["model2"]

    threading.Thread(target=train.train_model, args=("model1", model1_config['filters'], config['batch_size'], config['epochs'],  model1_config['optimizer'])).start()
    threading.Thread(target=train.train_model, args=("model2", model2_config['filters'], config['batch_size'], config['epochs'],  model2_config['optimizer'])).start()

    return jsonify({'status': 'Training started'})

@app.route('/predict_samples', methods=['POST'])
def predict_samples():
    model_name = request.json['model']
    
    # Load the trained model
    model_path = f'models/{model_name}.pth'
    if not os.path.exists(model_path):
        return jsonify({'error': 'Model not found'}), 404

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model configuration
    with open(f'static/losses_{model_name}.json', 'r') as f:
        config = json.load(f)
    
    model = FashionCNN(filters=config['config']).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Prepare test dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    # Get 10 random test images
    random_indices = random.sample(range(len(test_dataset)), 10)
    sample_results = []

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    with torch.no_grad():
        for idx in random_indices:
            image, label = test_dataset[idx]
            image = image.unsqueeze(0).to(device)
            output = model(image)
            _, predicted = torch.max(output.data, 1)
            
            # Save the image
            image_path = f'static/sample_{model_name}_{idx}.png'
            print(image_path)
            plt.switch_backend('Agg') 
            #train.save_image(test_dataset[idx][0], image_path)
            plt.figure(figsize=(2, 2))
            plt.imshow(test_dataset[idx][0].squeeze(), cmap='gray')
            plt.axis('off')
            plt.savefig(image_path)
            plt.close()
            
            sample_results.append({
                'image_path': image_path,
                'true_label': class_names[label],
                'predicted_label': class_names[predicted.item()]
            })

    return jsonify(sample_results)

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

if __name__ == '__main__':
    app.run(debug=False)  # Set debug to False to reduce logging