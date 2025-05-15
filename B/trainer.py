"""
BiRNN-Attention Model Trainer for Racial Bias Detection
=======================================================
This module implements the training and evaluation of the
BiRNN with Attention model for detecting racial bias in text.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# Generate training progress plot
from B.plots import plot_training_progress, plot_confusion_matrix, plot_category_confusion_matrix

# Import the model and dataset from model.py
from B.model import BiRNNAttention, TextDataset

# Define device for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define output directory
DATA_DIR = "Datasets/hatexplain_data"
OUTPUT_DIR = os.path.join("outputs")

# Trainer class 
class B:
    def __init__(self, args=None):
        """
        Initialize the BiRNN with Attention model for racial bias detection
        """
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.model = None
        self.binary_model_path = os.path.join(OUTPUT_DIR, "model_B", "binary_model.pt")
        self.multitask_model_path = os.path.join(OUTPUT_DIR, "model_B", "multitask_model.pt")
        self.best_model_path = os.path.join(OUTPUT_DIR, "model_B", "best_model.pt")

        # Create main output directory
        os.makedirs(os.path.join(OUTPUT_DIR, "model_B"), exist_ok=True)

        # Create subdirectories for plots and results
        os.makedirs(os.path.join(OUTPUT_DIR, "model_B", "binary"), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, "model_B", "multitask"), exist_ok=True)

    def prepare_dataloader(self, df, batch_size=32, multi_task=False):
        """
        Prepare dataloader for training or evaluation
        """
        # Make a copy to avoid modifying the original dataframe
        df_copy = df.copy()

        if multi_task and 'racial_category' in df_copy.columns:
            # For multi-task, filter out NaNs
            filtered_df = df_copy.dropna(subset=['racial_category'])

            if len(filtered_df) < len(df_copy):
                print(f"Filtered out {len(df_copy) - len(filtered_df)} rows with NaN racial_category for multi-task")

            # Encode category labels using filtered data
            if not hasattr(self, 'category_encoder'):
                self.category_encoder = LabelEncoder()
                self.category_encoder.fit(filtered_df['racial_category'].unique())

                # Save category labels
                self.category_labels = self.category_encoder.classes_
                print(f"Category labels: {self.category_labels}")

            category_labels = self.category_encoder.transform(filtered_df['racial_category'])

            # Use filtered data for multi-task
            dataset = TextDataset(
                filtered_df['text'].tolist(),
                filtered_df['label'].tolist(),
                self.tokenizer,
                category_labels.tolist()
            )
        else:
            # For binary-only, use the full dataset
            dataset = TextDataset(
                df_copy['text'].tolist(),
                df_copy['label'].tolist(),
                self.tokenizer,
                None
            )

        return DataLoader(dataset, batch_size=batch_size, shuffle=(df_copy['split'] == 'train').any())

    def train(self, train_df, val_df):
        """
        Train both binary and multi-task models separately
        Args:
            train_df (pd.DataFrame): Training data
            val_df (pd.DataFrame): Validation data
            args: Additional arguments (not used)
        """
        # train the binary model using ALL data
        print("\n==== Training Binary Classification Model (ALL data) ====")
        binary_accuracy = self.train_model(train_df, val_df, multi_task=False,
                                        model_path=self.binary_model_path)

        #if multi-task is possible, train the multi-task model with filtered data
        if 'racial_category' in train_df.columns:
            print("\n==== Training Multi-task Model (filtered data) ====")
            multitask_accuracy = self.train_model(train_df, val_df, multi_task=True,
                                                model_path=self.multitask_model_path)

            # Set the multitask model as the default best model
            torch.save(torch.load(self.multitask_model_path), self.best_model_path)
            print(f"Multi-task model saved as best model with accuracy: {multitask_accuracy:.4f}")

            # Return a dictionary with both accuracies
            return {'binary': binary_accuracy, 'multitask': multitask_accuracy}
        else:
            # Set binary model as the best model
            torch.save(torch.load(self.binary_model_path), self.best_model_path)

            # Return just the binary accuracy
            return binary_accuracy

    def train_model(self, train_df, val_df, multi_task=False, model_path=None):
        """
        Train a single model (binary or multi-task)
        Args:
            train_df (pd.DataFrame): Training data
            val_df (pd.DataFrame): Validation data
            multi_task (bool): Whether to train a multi-task model
            model_path (str): Path to save the trained model
        """
        if multi_task:
            print("Training BiRNN with Attention model for multi-task learning...")
        else:
            print("Training BiRNN with Attention model for binary classification only...")

        # Create dataloaders
        train_loader = self.prepare_dataloader(train_df, batch_size=32, multi_task=multi_task)
        val_loader = self.prepare_dataloader(val_df, batch_size=32, multi_task=multi_task)

        # Number of categories
        num_categories = len(self.category_encoder.classes_) if multi_task else None

        # Initialize metrics tracking
        # Track appropriate accuracy based on model type
        if multi_task:
            # For multi-task, track category accuracy
            cat_train_acc_history = []
            cat_val_acc_history = []
        else:
            # For binary, track binary accuracy
            bi_train_acc_history = []
            bi_val_acc_history = []

        train_loss_history = []
        val_loss_history = []

        # Initialize model
        model = BiRNNAttention(
            vocab_size=self.tokenizer.vocab_size,
            embedding_dim=100,
            hidden_dim=64,
            output_dim=2,
            n_layers=1,
            bidirectional=True,
            dropout=0.6,
            pad_idx=self.tokenizer.pad_token_id,
            num_categories=num_categories
        ).to(device)

        # Store as the current model
        self.model = model

        # Define optimizer and loss function TODO: play around with parameters
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        binary_criterion = nn.CrossEntropyLoss()
        category_criterion = nn.CrossEntropyLoss() if multi_task else None

        # Training loop
        n_epochs = 20
        best_val_accuracy = 0.0
        best_model_state = None

        # Add early stopping
        patience = 3
        epochs_without_improvement = 0

        for epoch in range(n_epochs):
            # Training
            model.train()
            epoch_loss = 0
            binary_preds = []
            binary_labels = []
            if multi_task:
                category_preds = []
                category_labels = []

            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs} [Train]"):
                optimizer.zero_grad()

                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                binary_label = batch['binary_labels'].to(device)

                # Calculate text lengths from attention mask
                text_lengths = attention_mask.sum(dim=1)

                # Forward pass - different handling for multi-task
                if multi_task:
                    category_label = batch['category_labels'].to(device)
                    binary_output, category_output = model(input_ids, text_lengths)

                    # Calculate losses
                    binary_loss = binary_criterion(binary_output, binary_label)
                    category_loss = category_criterion(category_output, category_label)

                    # Combined loss
                    loss = binary_loss + category_loss

                    # Get predictions
                    binary_pred = torch.argmax(binary_output, dim=1)
                    category_pred = torch.argmax(category_output, dim=1)

                    # Store for metrics
                    category_preds.extend(category_pred.cpu().numpy())
                    category_labels.extend(category_label.cpu().numpy())
                else:
                    output = model(input_ids, text_lengths)
                    loss = binary_criterion(output, binary_label)
                    binary_pred = torch.argmax(output, dim=1)

                # Backward pass
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

                # Store predictions for accuracy calculation
                if multi_task:
                    category_preds.extend(category_pred.cpu().numpy())
                    category_labels.extend(category_label.cpu().numpy())
                else:
                    binary_preds.extend(binary_pred.cpu().numpy())
                    binary_labels.extend(binary_label.cpu().numpy())

            train_loss = epoch_loss / len(train_loader)

            # After calculating metrics
            train_loss_history.append(train_loss)

            print(f"Epoch {epoch+1}/{n_epochs}:")
            print(f" Loss: {train_loss:.4f}")
            # Process results conditionally based on model type
            if multi_task:
                # For multi-task, report and track category accuracy
                category_accuracy = accuracy_score(category_labels, category_preds)
                cat_train_acc_history.append(category_accuracy)
                print(f" Category Train Accuracy: {category_accuracy:.4f}")
            else:
                # For binary, report and track binary metrics
                bi_train_accuracy = accuracy_score(binary_labels, binary_preds)
                bi_train_f1 = f1_score(binary_labels, binary_preds)
                bi_train_acc_history.append(bi_train_accuracy)
                print(f" Binary Train Accuracy: {bi_train_accuracy:.4f}")
                print(f" Binary Train F1: {bi_train_f1:.4f}")

            # Validation ===============================================================
            model.eval()
            val_loss = 0
            binary_preds = []
            binary_labels = []
            if multi_task:
                category_preds = []
                category_labels = []

            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{n_epochs} [Val]"):
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    binary_label = batch['binary_labels'].to(device)

                    text_lengths = attention_mask.sum(dim=1)
                    # Forward pass - different handling for multi-task
                    if multi_task:
                        category_label = batch['category_labels'].to(device)
                        binary_output, category_output = model(input_ids, text_lengths)

                        # Calculate losses
                        binary_loss = binary_criterion(binary_output, binary_label)
                        category_loss = category_criterion(category_output, category_label)

                        # Combined loss
                        loss = binary_loss + category_loss

                        # Get predictions
                        binary_pred = torch.argmax(binary_output, dim=1)
                        category_pred = torch.argmax(category_output, dim=1)

                        # Store for metrics
                        category_preds.extend(category_pred.cpu().numpy())
                        category_labels.extend(category_label.cpu().numpy())
                    else:
                        output = model(input_ids, text_lengths)
                        loss = binary_criterion(output, binary_label)
                        binary_pred = torch.argmax(output, dim=1)

                    val_loss += loss.item()

                    # Store for metrics
                    binary_preds.extend(binary_pred.cpu().numpy())
                    binary_labels.extend(binary_label.cpu().numpy())

            val_loss = val_loss / len(val_loader)
            val_loss_history.append(val_loss)
            print(f" Val Loss: {val_loss:.4f}")

            if multi_task:
                val_category_accuracy = accuracy_score(category_labels, category_preds)
                cat_val_acc_history.append(val_category_accuracy)
                print(f" Val Category Accuracy: {val_category_accuracy:.4f}")

                # Use category accuracy for model selection and early stopping
                if val_category_accuracy > best_val_accuracy:
                    best_val_accuracy = val_category_accuracy
                    best_model_state = model.state_dict()
                    epochs_without_improvement = 0
                    print(f" Saved new best model with category validation accuracy: {val_category_accuracy:.4f}")
                else:
                    epochs_without_improvement += 1
            else:
                bi_val_accuracy = accuracy_score(binary_labels, binary_preds)
                bi_val_f1 = f1_score(binary_labels, binary_preds)
                bi_val_acc_history.append(bi_val_accuracy)

                print(f" Binary Val Accuracy: {bi_val_accuracy:.4f}")
                print(f" Binary Val F1: {bi_val_f1:.4f}")

                # For binary-only, use binary accuracy
                if bi_val_accuracy > best_val_accuracy:
                    best_val_accuracy = bi_val_accuracy
                    best_model_state = model.state_dict()
                    epochs_without_improvement = 0
                    print(f" Saved new best model with binary validation accuracy: {bi_val_accuracy:.4f}")
                else:
                    epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

        # Define directory for plots
        plot_dir = os.path.join(OUTPUT_DIR, "model_B", "binary" if not multi_task else "multitask")

        if multi_task:
            # For multi-task, plot category accuracy
            plot_training_progress(
                cat_train_acc_history,
                cat_val_acc_history,
                train_loss_history,
                val_loss_history,
                plot_dir,
            )
        else:
            # For binary, plot binary accuracy
            plot_training_progress(
                bi_train_acc_history,
                bi_val_acc_history,
                train_loss_history,
                val_loss_history,
                plot_dir
            )

        # Save the best model
        torch.save(best_model_state, model_path)
        print(f"Best model saved to {model_path}")

        # Load the best model
        model.load_state_dict(best_model_state)

        # Return the best validation accuracy
        return best_val_accuracy

    def test(self, test_df):
        """
        Test both binary and multi-task models
        """
        results = {}

        # Test binary model on ALL data
        print("\n==== Testing Binary Classification Model (ALL data) ====")
        binary_accuracy = self.test_model(test_df, self.binary_model_path, multi_task=False)
        results['binary'] = binary_accuracy

        # Test multi-task model if available
        if 'racial_category' in test_df.columns and os.path.exists(self.multitask_model_path):
            print("\n==== Testing Multi-task Model (filtered data) ====")
            multitask_accuracy, category_accuracy = self.test_model(
                test_df, self.multitask_model_path, multi_task=True)
            results['multitask_binary'] = multitask_accuracy
            results['category'] = category_accuracy

        return results

    def test_model(self, test_df, model_path, multi_task=False):
        """
        Test a specific model
        """
        # Load the model
        num_categories = len(self.category_encoder.classes_) if multi_task else None
        model = BiRNNAttention(
            vocab_size=self.tokenizer.vocab_size,
            embedding_dim=100,
            hidden_dim=64,
            output_dim=2,
            n_layers=1,
            bidirectional=True,
            dropout=0.6,
            pad_idx=self.tokenizer.pad_token_id,
            num_categories=num_categories
        ).to(device)
        model.load_state_dict(torch.load(model_path))
        model.eval()

        # Store as current model
        self.model = model

        # Prepare dataloader
        test_loader = self.prepare_dataloader(test_df, batch_size=32, multi_task=multi_task)

        # Define loss functions
        binary_criterion = nn.CrossEntropyLoss()
        category_criterion = nn.CrossEntropyLoss() if multi_task else None

        # Evaluation variables
        test_loss = 0
        binary_preds = []
        binary_labels = []
        if multi_task:
            category_preds = []
            category_labels = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Testing {'Multi-task' if multi_task else 'Binary'} model"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                binary_label = batch['binary_labels'].to(device)

                text_lengths = attention_mask.sum(dim=1)

                # Forward pass - different handling for multi-task
                if multi_task:
                    category_label = batch['category_labels'].to(device)
                    binary_output, category_output = model(input_ids, text_lengths)

                    # Calculate losses
                    binary_loss = binary_criterion(binary_output, binary_label)
                    category_loss = category_criterion(category_output, category_label)

                    # Combined loss
                    loss = binary_loss + category_loss

                    # Get predictions
                    binary_pred = torch.argmax(binary_output, dim=1)
                    category_pred = torch.argmax(category_output, dim=1)

                    # Store for metrics
                    category_preds.extend(category_pred.cpu().numpy())
                    category_labels.extend(category_label.cpu().numpy())
                else:
                    output = model(input_ids, text_lengths)
                    loss = binary_criterion(output, binary_label)
                    binary_pred = torch.argmax(output, dim=1)

                test_loss += loss.item()

                # Store for metrics
                binary_preds.extend(binary_pred.cpu().numpy())
                binary_labels.extend(binary_label.cpu().numpy())

        test_loss = test_loss / len(test_loader)
        bi_test_accuracy = accuracy_score(binary_labels, binary_preds)
        bi_test_f1 = f1_score(binary_labels, binary_preds)

        print(f"Test Loss: {test_loss:.4f}")
        print(f"Binary Test Accuracy: {bi_test_accuracy:.4f}")
        print(f"Binary Test F1 Score: {bi_test_f1:.4f}")

        # Save binary confusion matrix with simplified path
        model_type = "binary" if not multi_task else "multitask"
        confusion_path = os.path.join(OUTPUT_DIR, "model_B", f"{model_type}_confusion.png")

        # Plot and save confusion matrix
        plot_confusion_matrix(
            binary_labels,
            binary_preds,
            confusion_path
        )

        # For multi-task, also report category accuracy
        if multi_task:
            test_category_accuracy = accuracy_score(category_labels, category_preds)
            print(f"Category Test Accuracy: {test_category_accuracy:.4f}")

            # Get category names for report
            category_names = self.category_encoder.classes_.tolist()

            # Generate detailed category classification report
            test_category_report = classification_report(
                category_labels,
                category_preds,
                target_names=category_names,
                output_dict=True
            )

            # Save category results
            with open(os.path.join(OUTPUT_DIR, "model_B", "category_test_results.json"), 'w') as f:
                json.dump(test_category_report, f, indent=2)

            # Plot category confusion matrix with simplified path
            cat_confusion_path = os.path.join(OUTPUT_DIR, "model_B", "category_confusion.png")

            # Plot and save category confusion matrix
            plot_category_confusion_matrix(
                category_labels,
                category_preds,
                self.category_encoder.classes_,
                cat_confusion_path
            )

            return bi_test_accuracy, test_category_accuracy

        # Save binary test results
        test_report = classification_report(binary_labels, binary_preds, output_dict=True)
        with open(os.path.join(OUTPUT_DIR, "model_B", f"{model_type}_test_results.json"), 'w') as f:
            json.dump(test_report, f, indent=2)

        return bi_test_accuracy