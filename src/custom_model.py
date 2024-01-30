import torch.nn as nn
from transformers import AutoModel
import torch

class DAIGTModel(nn.Module):
    def __init__(self, model_path, config, tokenizer, pretrained=False):
        """
        Initialize the DAIGTModel.

        Args:
            model_path (str): Path to the pre-trained model.
            config (AutoConfig): Configuration for the model.
            tokenizer (AutoTokenizer): Tokenizer for text encoding.
            pretrained (bool, optional): Whether to load pre-trained weights. Defaults to False.
        """
        super().__init__()
        if pretrained:
            self.model = AutoModel.from_pretrained(model_path, config=config)
        else:
            self.model = AutoModel.from_config(config)
        self.classifier = nn.Linear(config.hidden_size, 1)   

    def forward_features(self, input_ids, attention_mask=None):
        """
        Forward pass through the model to get the features.

        Args:
            input_ids (tensor): Input tensor of tokenized input IDs.
            attention_mask (tensor, optional): Attention mask tensor. Defaults to None.

        Returns:
            tensor: The extracted features.
        """
        outputs = self.model(input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        embeddings = sum_embeddings / sum_mask
        return embeddings

    def forward(self, input_ids, attention_mask):
        """
        Forward pass through the model.

        Args:
            input_ids (tensor): Input tensor of tokenized input IDs.
            attention_mask (tensor): Attention mask tensor.

        Returns:
            tensor: The output logits.
        """
        embeddings = self.forward_features(input_ids, attention_mask)
        logits = self.classifier(embeddings)
        return logits