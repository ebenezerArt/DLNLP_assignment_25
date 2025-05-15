"""
Racial Bias Dataset Processing with Categories
=============================================
This script processes the HateXplain dataset to focus on racial bias, creating
a multi-class categorization system for different types of racial bias.
"""

import json
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import random

# Download necessary NLTK resources
nltk.download('stopwords') # filter out irrelevant words for computational efficiency
nltk.download('wordnet')  # lemmatization - grouping words with similar meanings

# Define the main data directory
DATA_DIR = "Datasets/hatexplain_data"
OUTPUT_DIR = os.path.join(DATA_DIR, "racial_bias")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define racial target categories and their associated terms
RACIAL_CATEGORIES = {
    "African": ["african", "black", "negro", "n*gger", "nigger", "nigga"],
    "Asian": ["asian", "chinese", "japanese", "korean", "oriental", "ch*nk", "chink"],
    "Hispanic": ["hispanic", "latino", "latina", "mexican", "spic", "sp*c"],
    "Muslim/Arab": ["muslim", "arab", "islamic", "middle eastern", "terrorist"],
    "Jewish": ["jewish", "jew", "judaism", "semitic", "antisemitic"],
    "White": ["white", "caucasian", "cracker", "colonizer", "gringo"],
    "General": ["race", "racial", "racist", "ethnicity", "ethnic", "poc", "people of color"]
}

# Define types of racial bias
BIAS_TYPES = {
    "Stereotyping": ["stereotype", "all", "they are", "always", "typical"],
    "Slurs": [], # This will be populated from the racial categories
    "Ideological": ["supremacy", "supremacist", "nationalist", "race war", "genocide"],
    "Discrimination": ["go back", "don't belong", "not welcome", "immigrant", "deport"],
    "Othering": ["them", "they", "these people", "those people", "you people"],
    "None": []  # For posts without racial bias
}

# Populate slurs from the racial categories
for category, terms in RACIAL_CATEGORIES.items():
    for term in terms:
        if '*' in term or term in ["nigger", "nigga", "chink", "spic"]:
            BIAS_TYPES["Slurs"].append(term)

def categorize_racial_bias(text, targets, rationales, tokens):
    """
    Categorize the type of racial bias in a post

    Args:
        text: The full text of the post
        targets: List of targets from annotators
        rationales: List of rationales (highlighted portions)
        tokens: List of tokens in the text

    Returns:
        racial_category: The racial group being targeted
        bias_type: The type of bias exhibited
    """
    #make all text lower case for easier matching
    text_lower = text.lower()

    # Extract highlighted text from annotated rationales
    highlighted_texts = []
    for rationale_set in rationales:
        highlighted = [tokens[i] for i, val in enumerate(rationale_set) if val == 1]
        if highlighted:
            highlighted_texts.append(" ".join(highlighted))

    highlighted_text = " ".join(highlighted_texts).lower() if highlighted_texts else ""

    # Determine racial category
    racial_category = "General"  # Default category
    category_scores = {category: 0 for category in RACIAL_CATEGORIES.keys()}

    for category, terms in RACIAL_CATEGORIES.items():
        for term in terms:
            if term in text_lower:
                category_scores[category] += 1
                # Give more weight if the term appears in highlighted text or targets
                if term in highlighted_text:
                    category_scores[category] += 2
                if any(term in target.lower() for target in targets if target != "None"):
                    category_scores[category] += 3

    # Determine the racial category with the highest score
    if any(score > 0 for score in category_scores.values()):
        racial_category = max(category_scores.items(), key=lambda x: x[1])[0]

    # Determine bias type
    bias_type = "None"  # Default type
    type_scores = {btype: 0 for btype in BIAS_TYPES.keys()}

    for btype, terms in BIAS_TYPES.items():
        for term in terms:
            if term in text_lower:
                type_scores[btype] += 1
                # Give more weight if the term appears in highlighted text
                if term in highlighted_text:
                    type_scores[btype] += 2

    # Determine the bias type with the highest score
    if any(score > 0 for score in type_scores.values()):
        bias_type = max(type_scores.items(), key=lambda x: x[1])[0]

    # If no specific bias type is detected but racial terms exist, default to "Stereotyping"
    if bias_type == "None" and racial_category != "General":
        bias_type = "Stereotyping"

    return racial_category, bias_type

def load_and_categorize_data():
    """Load the dataset from the formatted JSON files and categorize racial bias types"""

    print("Loading and categorizing data...")

    # List all formatted data files
    data_paths = [
        os.path.join(DATA_DIR, "train_formatted.json"),
        os.path.join(DATA_DIR, "val_formatted.json"),
        os.path.join(DATA_DIR, "test_formatted.json")
    ]

    all_data = {}
    racial_data = {}
    non_racial_data = {}

    # Process each data file
    for data_path in data_paths:
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            split_name = os.path.basename(data_path).split("_")[0]
            print(f"Processing {split_name} split...")

            # Process instances for this split
            racial_instances = {}
            non_racial_instances = {}

            for post_id, post_data in tqdm(data.items()):
                # Extract text from tokens
                text = " ".join(post_data["post_tokens"])

                # Extract majority label through voting
                labels = [anno["label"] for anno in post_data["annotators"]]
                label_counter = Counter(labels)
                majority_label = label_counter.most_common(1)[0][0]

                # Extract all targets mentioned by annotators
                all_targets = []
                for anno in post_data["annotators"]:
                    all_targets.extend(anno["target"])

                # Remove duplicates but preserve order
                unique_targets = []
                for target in all_targets:
                    if target not in unique_targets:
                        unique_targets.append(target)

                # Check for racial bias and categorize
                has_racial_target = False
                racial_category = "None"
                bias_type = "None"

                # Check if any racial targets are present
                for target in unique_targets:
                    if target != "None":
                        for category, terms in RACIAL_CATEGORIES.items():
                            if any(term in target.lower() for term in terms):
                                has_racial_target = True
                                break

                # If post has racial targets, categorize the bias
                if has_racial_target:
                    racial_category, bias_type = categorize_racial_bias(
                        text, unique_targets, post_data["rationales"], post_data["post_tokens"]
                    )

                # Add to all data dictionary with category information
                all_data[post_id] = {
                    "post_id": post_id,
                    "text": text,
                    "post_tokens": post_data["post_tokens"],
                    "annotators": post_data["annotators"],
                    "rationales": post_data["rationales"],
                    "has_racial_target": has_racial_target,
                    "racial_category": racial_category,
                    "bias_type": bias_type,
                    "label": majority_label,
                    "targets": unique_targets,
                    "split": split_name
                }

                # Add to appropriate dictionary
                if has_racial_target:
                    racial_instances[post_id] = all_data[post_id]
                else:
                    non_racial_instances[post_id] = all_data[post_id]

            racial_data[split_name] = racial_instances
            non_racial_data[split_name] = non_racial_instances

            print(f"Found {len(racial_instances)} posts with racial bias in {split_name} split")
            print(f"Found {len(non_racial_instances)} posts without racial bias in {split_name} split")

        except FileNotFoundError:
            print(f"Warning: File not found: {data_path}")

    # Create balanced dataset with equal representation of racial and non-racial posts
    balanced_data = {}

    for split in racial_data.keys():
        racial_count = len(racial_data[split])
        non_racial_posts = list(non_racial_data[split].keys())

        # If not enough non-racial posts, take all available
        if len(non_racial_posts) > racial_count:
            sampled_non_racial = random.sample(non_racial_posts, racial_count)
        else:
            sampled_non_racial = non_racial_posts

        # Combine racial and sampled non-racial posts
        balanced_split = {}
        for post_id in racial_data[split]:
            balanced_split[post_id] = all_data[post_id]

        for post_id in sampled_non_racial:
            balanced_split[post_id] = all_data[post_id]

        balanced_data[split] = balanced_split

        print(f"Created balanced {split} split with {len(balanced_split)} posts "
              f"({len(racial_data[split])} racial, {len(sampled_non_racial)} non-racial)")

    # Save the categorized data
    with open(os.path.join(OUTPUT_DIR, "categorized_data.json"), 'w') as f:
        json.dump(all_data, f, indent=2)

    with open(os.path.join(OUTPUT_DIR, "balanced_categorized_data.json"), 'w') as f:
        json.dump(balanced_data, f, indent=2)

    return all_data, balanced_data

def analyze_categories(balanced_data):
    """Analyze the distribution of racial categories and bias types"""

    print("\nAnalyzing racial categories and bias types...")

    # Initialize counters
    racial_category_counts = Counter()
    bias_type_counts = Counter()
    category_bias_matrix = {cat: Counter() for cat in RACIAL_CATEGORIES.keys()}
    category_bias_matrix["None"] = Counter()

    # Count instances by category and type
    for split, instances in balanced_data.items():
        for post_id, data in instances.items():
            if data["has_racial_target"]:
                racial_category_counts[data["racial_category"]] += 1
                bias_type_counts[data["bias_type"]] += 1
                category_bias_matrix[data["racial_category"]][data["bias_type"]] += 1
            else:
                racial_category_counts["None"] += 1
                bias_type_counts["None"] += 1
                category_bias_matrix["None"]["None"] += 1

    # Print statistics
    print("\nRacial Category Distribution:")
    for category, count in racial_category_counts.most_common():
        print(f"  - {category}: {count} posts")

    print("\nBias Type Distribution:")
    for bias_type, count in bias_type_counts.most_common():
        print(f"  - {bias_type}: {count} posts")

    # Create visualizations
    create_category_visualizations(racial_category_counts, bias_type_counts, category_bias_matrix)

    return racial_category_counts, bias_type_counts, category_bias_matrix

def create_category_visualizations(racial_category_counts, bias_type_counts, category_bias_matrix):
    """Create visualizations for racial categories and bias types"""

    print("\nCreating category visualizations...")

    # Racial category distribution
    plt.figure(figsize=(12, 6))
    category_df = pd.DataFrame({
        'Category': list(racial_category_counts.keys()),
        'Count': list(racial_category_counts.values())
    }).sort_values('Count', ascending=False)

    sns.barplot(x='Count', y='Category', data=category_df)
    plt.title('Distribution of Racial Categories')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "racial_category_distribution.png"))

    # Bias type distribution
    plt.figure(figsize=(12, 6))
    bias_df = pd.DataFrame({
        'Bias Type': list(bias_type_counts.keys()),
        'Count': list(bias_type_counts.values())
    }).sort_values('Count', ascending=False)

    sns.barplot(x='Count', y='Bias Type', data=bias_df)
    plt.title('Distribution of Bias Types')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "bias_type_distribution.png"))

    # Category-Bias matrix heatmap
    # Convert the nested counters to a DataFrame
    matrix_data = []
    for category, bias_counts in category_bias_matrix.items():
        for bias_type, count in bias_counts.items():
            matrix_data.append({
                'Category': category,
                'Bias Type': bias_type,
                'Count': count
            })

    matrix_df = pd.DataFrame(matrix_data)
    pivot_df = matrix_df.pivot(index='Category', columns='Bias Type', values='Count').fillna(0)

    plt.figure(figsize=(14, 10))
    sns.heatmap(pivot_df, annot=True, fmt='g', cmap='viridis')
    plt.title('Distribution of Bias Types by Racial Category')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "category_bias_matrix.png"))

def process_for_multi_class_modeling(balanced_data):
    """Prepare the data for multi-class classification modeling"""

    print("\nPreprocessing data for multi-class models...")

    # Initialize lemmatizer and stopwords
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    def clean_text(text):
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and numbers
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        # Tokenize and remove stopwords
        tokens = [lemmatizer.lemmatize(word) for word in text.split()
                  if word not in stop_words and len(word) > 2]
        return ' '.join(tokens)

    # Prepare data for different classification tasks
    binary_classification = []  # Racial bias detection (binary)
    category_classification = []  # Racial category classification
    type_classification = []  # Bias type classification
    multi_classification = []  # Combined category and type

    for split, instances in balanced_data.items():
        for post_id, data in instances.items():
            # Clean the text
            cleaned_text = clean_text(data["text"])

            # Binary classification: 1 if racial bias, 0 if not
            binary_classification.append({
                "text": data["text"],
                "cleaned_text": cleaned_text,
                "label": 1 if data["has_racial_target"] else 0,
                "split": split,
                "post_id": post_id
            })

            # Category classification
            # Category classification: ONLY add if racial_category is not NaN
            if data["has_racial_target"] and pd.notna(data["racial_category"]):
                category_classification.append({
                    "text": data["text"],
                    "cleaned_text": cleaned_text,
                    "label": data["racial_category"],
                    "split": split,
                    "post_id": post_id
                })

            # Type classification
            type_classification.append({
                "text": data["text"],
                "cleaned_text": cleaned_text,
                "label": data["bias_type"],
                "split": split,
                "post_id": post_id
            })

            # Multi-task classification: ONLY include valid racial categories
            # This keeps all data for binary classification but filters racial categories
            if pd.notna(data["racial_category"]):
                multi_classification.append({
                    "text": data["text"],
                    "cleaned_text": cleaned_text,
                    "label": 1 if data["has_racial_target"] else 0,  # Binary label
                    "racial_category": data["racial_category"],      # Only valid categories
                    "split": split,
                    "post_id": post_id
                })
            else:
                # Include for binary classification only (no racial_category)
                multi_classification.append({
                    "text": data["text"],
                    "cleaned_text": cleaned_text,
                    "label": 1 if data["has_racial_target"] else 0,  # Binary label
                    # NO racial_category field for these
                    "split": split,
                    "post_id": post_id
                })


    # Convert to DataFrames
    binary_df = pd.DataFrame(binary_classification)
    category_df = pd.DataFrame(category_classification)
    type_df = pd.DataFrame(type_classification)
    multi_df = pd.DataFrame(multi_classification)

    # Save processed data
    binary_df.to_csv(os.path.join(OUTPUT_DIR, "binary_classification.csv"), index=False)
    category_df.to_csv(os.path.join(OUTPUT_DIR, "category_classification.csv"), index=False)
    type_df.to_csv(os.path.join(OUTPUT_DIR, "type_classification.csv"), index=False)
    multi_df.to_csv(os.path.join(OUTPUT_DIR, "multi_classification.csv"), index=False)

    # Create label maps for each classification task
    binary_labels = {0: "No Racial Bias", 1: "Racial Bias"}
    category_labels = {cat: cat for cat in RACIAL_CATEGORIES.keys()}
    category_labels["None"] = "None"
    type_labels = {btype: btype for btype in BIAS_TYPES.keys()}

    # Count unique multi labels and create map
    multi_label_counts = Counter(multi_df["label"])
    multi_labels = {label: label for label in multi_label_counts.keys()}

    # Save label maps
    label_maps = {
        "binary": binary_labels,
        "category": category_labels,
        "type": type_labels,
        "multi": multi_labels
    }

    with open(os.path.join(OUTPUT_DIR, "label_maps.json"), 'w') as f:
        json.dump(label_maps, f, indent=2)

    print(f"Processed data saved to {OUTPUT_DIR}")

    # Print dataset statistics
    print("\nDataset statistics:")
    print(f"Binary classification: {len(binary_df)} samples")
    print(f"  - Racial bias: {sum(binary_df['label'])} samples")
    print(f"  - No racial bias: {len(binary_df) - sum(binary_df['label'])} samples")

    print(f"\nCategory classification: {len(category_df)} samples")
    for category, count in Counter(category_df["label"]).most_common():
        print(f"  - {category}: {count} samples")

    print(f"\nBias type classification: {len(type_df)} samples")
    for btype, count in Counter(type_df["label"]).most_common():
        print(f"  - {btype}: {count} samples")

    print(f"\nMulti classification: {len(multi_df)} samples")
    print(f"  - Unique labels: {len(multi_label_counts)}")

    return binary_df, category_df, type_df, multi_df, label_maps




# def main():
#     """Main execution function"""

#     # Create output directory if it doesn't exist
#     os.makedirs(OUTPUT_DIR, exist_ok=True)

#     # Step 1: Load and categorize data
#     all_data, balanced_data = load_and_categorize_data()

#     # Step 2: Analyze categories
#     racial_category_counts, bias_type_counts, category_bias_matrix = analyze_categories(balanced_data)

#     # Step 3: Process for multi-class modeling
#     binary_df, category_df, type_df, multi_df, label_maps = process_for_multi_class_modeling(balanced_data)

#     print(f"\nAll processed data saved to {OUTPUT_DIR}")

# if __name__ == "__main__":
#     main()