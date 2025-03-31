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

1. Set up environment variables:

   - Copy `.env.example` to create your own `.env` file:
     ```bash
     cp .env.example .env
     ```
   - Edit `.env` with your paths:
     ```bash
     TRAIN_CSV_PATH=/path/to/train_imagenette.csv
     VALIDATION_CSV_PATH=/path/to/val_imagenette.csv
     PATH_PREFIX=/path/to/imagenette2-320
     PATH_TO_EXAMPLE_IMAGE=/path/to/example/image.JPEG
     ```

2. Place the Imagenette dataset in your specified directory (PATH_PREFIX in .env)

3. Run the notebooks:

   - `multinomial_logistic_regression.ipynb` for the Softmax classifier
   - `convolutional_neural_network.ipynb` for the CNN implementation

4. Model outputs:
   - Trained models will be saved to the directory specified in your .env
   - Feature maps will be saved to the 'feature_maps' directory
   - Training logs and visualizations will be displayed in the notebooks

## Model Saving

Models are automatically saved with timestamps:

```python
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = f"cnn_model_{timestamp}.h5"
```

## Feature Map Visualization

The project includes utilities to visualize and save feature maps from convolutional layers, helping understand what patterns the model learns to detect.
