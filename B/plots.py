import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Set a clean, readable style
plt.style.use('seaborn-v0_8')

def plot_training_progress(train_acc, val_acc, train_loss, val_loss, save_dir):
    """Plot training progress (accuracy and loss) with improved formatting"""
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot accuracy
    epochs = range(len(train_acc))
    ax1.plot(epochs, train_acc, 'o-', color='#1f77b4', linewidth=2, label='Train')
    ax1.plot(epochs, val_acc, 'o-', color='#ff7f0e', linewidth=2, label='Validation')
    ax1.set_title('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)

    # Plot loss
    ax2.plot(epochs, train_loss, 'o-', color='#1f77b4', linewidth=2, label='Train')
    ax2.plot(epochs, val_loss, 'o-', color='#ff7f0e', linewidth=2, label='Validation')
    ax2.set_title('Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    # Set layout
    plt.suptitle('Model Training Progress', fontsize=14)
    plt.tight_layout()

    # Save figure
    plt.savefig(os.path.join(save_dir, 'training_progress.png'), dpi=200, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(y_true, y_pred, save_path):
    """Plot simple confusion matrix for binary classification"""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Racial Bias', 'Racial Bias'],
                yticklabels=['No Racial Bias', 'Racial Bias'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()

    # Save figure
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def plot_category_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Plot confusion matrix for categories with improved formatting"""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Category Confusion Matrix')

    # Rotate x labels if there are many classes
    if len(class_names) > 4:
        plt.xticks(rotation=45, ha='right')

    plt.tight_layout()

    # Save figure
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


# Additional utility function for comparing models (optional)
def compare_models(models_data, save_path, metric='accuracy'):
    """Plot comparison of training metrics across multiple models"""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(10, 6))

    # Plot data for each model
    for model_name, data in models_data.items():
        plt.plot(data['train'], '-o', linewidth=2, markersize=4, label=f'{model_name} (Train)')
        plt.plot(data['val'], '--o', linewidth=2, markersize=4, label=f'{model_name} (Val)')

    # Set labels and title
    plt.xlabel('Epoch')
    plt.ylabel(metric.capitalize())
    plt.title(f'Model {metric.capitalize()} Comparison')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Save figure
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()