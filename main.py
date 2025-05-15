import argparse
import gc
import torch
import os
import random
import numpy as np
import pandas as pd


# Import preprocessing module
from A.preprocessing import load_and_categorize_data, analyze_categories, process_for_multi_class_modeling
from B.trainer import B

# Define the main data directory
DATA_DIR = "Datasets/hatexplain_data"
OUTPUT_DIR = os.path.join(DATA_DIR, "racial_bias")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Check for GPU TODO: set cuda device as parser argument??
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def data_preprocessing():
    """
    Main data preprocessing function using your existing preprocessing script
    """
    print("Starting data preprocessing...")

    # Step 1: Load and categorize data
    all_data, balanced_data = load_and_categorize_data()

    # Step 2: Analyze categories (optional, but useful for understanding the data)
    # racial_category_counts, bias_type_counts, category_bias_matrix = analyze_categories(balanced_data)

    # Step 3: Process for multi-class modeling
    binary_df, category_df, type_df, multi_df, label_maps = process_for_multi_class_modeling(balanced_data)

    # Load the processed data for multi and binary classification
    # binary_df = pd.read_csv(os.path.join(OUTPUT_DIR, "binary_classification.csv"))
    multi_df = pd.read_csv(os.path.join(OUTPUT_DIR, "multi_classification.csv"))


    # Split into train, val, test (for binary classification)
    # train_df = binary_df[binary_df['split'] == 'train']
    # val_df = binary_df[binary_df['split'] == 'val']
    # test_df = binary_df[binary_df['split'] == 'test']

    # Split into train, val, test (for multi classification)
    train_df = multi_df[multi_df['split'] == 'train']
    val_df = multi_df[multi_df['split'] == 'val']
    test_df = multi_df[multi_df['split'] == 'test']

    print(f"Training data: {len(train_df)} samples")
    print(f"Validation data: {len(val_df)} samples")
    print(f"Test data: {len(test_df)} samples")

    return train_df, val_df, test_df



def main():

    # Initialize variables for results
    acc_train = acc_test = ''

    try:
        # Data preprocessing
        print("==================== Data Preprocessing ====================")
        data_train, data_val, data_test = data_preprocessing()

        print("Available columns:", data_train.columns.tolist())
        print("'racial_category' in columns:", 'racial_category' in data_train.columns)

        print("\n==================== BiRNN with Attention Model ====================")
        model = B()
        acc_train = model.train(data_train, data_val)
        acc_test = model.test(data_test)

        # Clean up memory
        del model
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    except Exception as e:
        print(f"An error occurred: {str(e)}")

    # For training accuracies
    if isinstance(acc_train, dict):
        # Use category accuracy for reporting when available
        train_acc_display = acc_train['multitask']
    else:
        #use binary
        train_acc_display = acc_train

    # For test accuracies
    if isinstance(acc_test, dict):
        # Format dictionary values to 4 decimal places
        acc_test_formatted = {k: f"{v:.4f}" for k, v in acc_test.items()}
    else:
        acc_test_formatted = acc_test

    print('Model:{:.4f},{};'.format(train_acc_display, acc_test_formatted))

if __name__ == "__main__":
    main()