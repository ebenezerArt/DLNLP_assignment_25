# Racial Bias Detection and Catergorisation with BiRNN-Attention Model

This repository contains code for detecting and categorizing racial bias in text using a Bidirectional RNN with Attention model. The implementation processes the HateXplain dataset, categorizing racial bias into multiple groups and employing a multi-task learning approach to both detect and categorize racial bias in text.

## Project Organization

The project is organized into the following directories:

- `A/` - Contains preprocessing logic for the HateXplain dataset
- `B/` - Contains the BiRNN-Attention model implementation and training code
- `Datasets/` - Contains scripts for downloading and preparing the HateXplain dataset (Also where HateXplain dataset is stored and preprocessed data is stored)
- `outputs/` - Contains trained models, confusion matrices, and training progress plots

## File Descriptions

### Main Files
- `main.py` - Entry point for the application, orchestrates data preprocessing and model training/testing
- `requirements.txt` - Lists all required Python packages

### Preprocessing (A/)
- `preprocessing.py` - Processes the HateXplain dataset to focus on racial bias, creating a multi-class categorization system

### Model Implementation (B/)
- `model.py` - Implements the BiRNN with Attention model architecture
- `trainer.py` - Contains the training and evaluation logic for the model
- `plots.py` - Utilities for creating visualizations of training progress and results

### Dataset (Datasets/)
- `DataDownload.py` - Downloads and formats the HateXplain dataset

## Required Packages

All required packages are listed in `requirements.txt` and include:

```
torch
transformers
datasets
scikit-learn
pandas
numpy
matplotlib
seaborn
nltk
tqdm
plots
```

## Installation Instructions

1. Clone this repository:
```bash
git clone https://github.com/ebenezerArt/DLNLP_assignment_25-21092873.git
cd ..\DLNLP_assignment_25-21092873
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Download and prepare the HateXplain dataset:
```bash
python Datasets/DataDownload.py
```

## Usage Instructions
### Training and Evaluation

To train and evaluate the BiRNN-Attention model:

```bash
python main.py
```

This script will:
1. Preprocess the data
2. Train both binary and multi-task models
3. Evaluate the models on the test set
4. Save the models and visualizations in the `outputs/model_B/` directory

## Model Outputs

After training, you'll find the following outputs in the `outputs/model_B/` directory:

- `binary_model.pt` - Trained model for binary racial bias detection
- `multitask_model.pt` - Trained model for multi-task (both detection and categorization)
- `best_model.pt` - The best-performing model saved
- `binary_confusion.png` - Confusion matrix for binary classification
- `category_confusion.png` - Confusion matrix for racial category classification
- `*_test_results.json` - Detailed test results
- Training progress visualizations in `binary/` and `multitask/` subdirectories

## Notes

- The models are trained using GPU if available, otherwise CPU
- Training parameters (epochs, learning rate, etc.) can be modified in the `B/trainer.py` file
- Early stopping is implemented to prevent overfitting
