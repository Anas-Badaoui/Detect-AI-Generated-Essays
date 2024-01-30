import torch
import os
from datasets import load_dataset
import matplotlib.pyplot as plt
import pandas as pd
from umap import UMAP
from sklearn.preprocessing import MinMaxScaler
import random
import numpy as np
import torch

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        """
        Reset the meter's values.
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        Update the meter's values.

        Args:
            val (float): The current value.
            n (int, optional): The number of samples. Defaults to 1.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def preprocess_dataset(path_raw_data, test_size=0.1, seed=1):
    if not os.path.exists(path_raw_data):
        raise FileNotFoundError(f"{path_raw_data} does not exist.")

    def class_encode(dataset):
        return dataset.class_encode_column("label")

    def split_dataset(dataset):
        return dataset.train_test_split(test_size=test_size, stratify_by_column="label", seed=seed)

    def clean_and_count(dataset):
        def clean_and_count_words_per_essay(example):
            example["text"] = [x.strip('\n') for x in example["text"]]
            example["Words Per Essay"] = [len(x.split()) for x in example["text"]]
            return example

        return dataset.map(clean_and_count_words_per_essay, batched=True)

    dataset = load_dataset("csv", data_files=[path_raw_data], split='train')
    dataset = class_encode(dataset)
    dataset = split_dataset(dataset)
    dataset = clean_and_count(dataset)

    return dataset

def plot_dataset(dataset):
    try:
        # Set the dataset format to pandas
        dataset.set_format(type='pandas')

        # Get the training data
        df = dataset['train'][:]

        # Create a figure with two subplots side by side
        fig, axs = plt.subplots(1, 2, figsize=(15, 7))

        # Customize the figure background color
        fig.patch.set_facecolor('white')

        # Plot the frequency of classes on the first subplot
        df["label"].value_counts(ascending=True).plot(kind='barh', ax=axs[0], color='skyblue', edgecolor='black')

        # Customize the first subplot
        axs[0].set_title("Frequency of Classes", fontsize=14)
        axs[0].set_xlabel('Frequency', fontsize=12)
        axs[0].set_ylabel('Class', fontsize=12)
        axs[0].grid(axis='x', linestyle='--', alpha=0.6)

        # Plot the boxplot of words per essay on the second subplot
        df.boxplot("Words Per Essay", by="label", grid=False, showfliers=False, color=dict(boxes='black', whiskers='black', medians='blue', caps='gray'), ax=axs[1])

        # Customize the second subplot
        axs[1].set_title('Words Per Essay by Label', fontsize=14)
        axs[1].set_xlabel('Label', fontsize=12)
        axs[1].set_ylabel('Words Per Essay', fontsize=12)
        axs[1].grid(axis='y', linestyle='--', alpha=0.6)

        # Display the figure
        plt.tight_layout()
        plt.show()

    except pd.errors.EmptyDataError:
        print("The dataset is empty.")
    except KeyError as e:
        print(f"The dataset does not contain a necessary column: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def generate_embeddings(X_train, y_train):
    # Scale features to [0,1] range
    X_scaled = MinMaxScaler().fit_transform(X_train)

    # Initialize and fit UMAP
    mapper = UMAP(n_components=2, metric="cosine").fit(X_scaled)

    # Create a DataFrame of 2D embeddings
    df_embed = pd.DataFrame(mapper.embedding_, columns=["X", "Y"])
    df_embed["label"] = y_train
    return df_embed

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True