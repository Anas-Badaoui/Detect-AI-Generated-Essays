import matplotlib.pyplot as plt
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

def plot_emb(df_emb):
    
    fig, axes = plt.subplots(1, 2, figsize=(7,5))
    axes = axes.flatten()
    cmaps = ["Blues", "Greens"]
    labels = ["Human", "AI"]

    for i, (label, cmap) in enumerate(zip(labels, cmaps)):
        df_emb_sub = df_emb.query(f"label == {i}")
        axes[i].hexbin(df_emb_sub["X"], df_emb_sub["Y"], cmap=cmap, gridsize=20, linewidths=(0,))
        axes[i].set_title(label)
        axes[i].set_xticks([]), axes[i].set_yticks([])
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_preds, y_true, labels):
    cm = confusion_matrix(y_true, y_preds, normalize="true")
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=False)
    plt.title("Normalized confusion matrix")
    plt.show()
    
def extract_hidden_states(batch, tokenizer, device, model):
    """
    Extracts the last hidden states from a batch of inputs.

    Args:
        batch (dict): A dictionary containing the input batch.

    Returns:
        dict: A dictionary containing the extracted hidden states.
    """
    inputs = {k:v.to(device) for k,v in batch.items() if k in tokenizer.model_input_names}
    
    with torch.no_grad():
        last_hidden_state = model(**inputs).last_hidden_state
    
    return {"hidden_state": last_hidden_state[:,0].cpu().numpy()}


def extract_hidden_states_custom_model(batch, tokenizer, device, model):
    """
    Extracts the last hidden states from a custom model for a given batch of inputs.

    Args:
        batch (dict): A dictionary containing the inputs for the model.

    Returns:
        dict: A dictionary containing the extracted hidden states for the [CLS] token.
    """
    # Place model inputs on the GPU
    inputs = {k:v.to(device) for k,v in batch.items() if k in tokenizer.model_input_names}
    encoder_block = model.model
    # Extract last hidden states
    with torch.no_grad():
        last_hidden_state = encoder_block(**inputs).last_hidden_state
    # Return vector for [CLS] token
    return {"hidden_state": last_hidden_state[:,0].cpu().numpy()}