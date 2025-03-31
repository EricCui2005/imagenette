# Image Classification Models

This repository contains implementations of different image classification models trained on the Imagenette dataset, a smaller subset of ImageNet. The project includes both a simple Softmax Classifier and a Convolutional Neural Network (CNN) implementation.

## Dataset

The project uses Imagenette2-320, which includes 10 classes:

- Tench
- English Springer
- Cassette Player
- Chain Saw
- Church
- French Horn
- Garbage Truck
- Gas Pump
- Golf Ball
- Parachute

## Models

### 1. Softmax Classifier

A simple linear classifier implemented using PyTorch that:

- Flattens input images to 1D arrays
- Uses a single fully connected layer
- Outputs class probabilities

### 2. Convolutional Neural Network

A more sophisticated model implemented using TensorFlow/Keras that:

- Processes 64x64 RGB images
- Includes convolutional layers
- Provides feature map visualization capabilities

## Features

- Image preprocessing and loading utilities
- Feature map visualization
- Model saving and loading functionality
- Training progress tracking
- Validation accuracy monitoring

## Installation

1. Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Place the Imagenette dataset in the `imagenette2-320/` directory
2. Run the notebooks:
   - `multinomial_logistic_regression.ipynb` for the Softmax classifier
   - `convolutional_neural_network.ipynb` for the CNN implementation

## Model Saving

Models are automatically saved with timestamps:

```python
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = f"cnn_model_{timestamp}.h5"
```

## Feature Map Visualization

The project includes utilities to visualize and save feature maps from convolutional layers, helping understand what patterns the model learns to detect.
