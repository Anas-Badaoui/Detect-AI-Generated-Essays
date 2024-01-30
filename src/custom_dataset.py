from torch.utils.data import Dataset
from transformers import AutoTokenizer

class DAIGTDataset(Dataset):
    def __init__(self, text_list, tokenizer, max_len, label_list):
        """
        Initialize the DAIGTDataset.

        Args:
            text_list (list): List of texts.
            tokenizer (AutoTokenizer): Tokenizer for text encoding.
            max_len (int): Maximum length of the input sequence.
            label_list (list): List of labels.
        """
        self.text_list=text_list
        self.tokenizer=tokenizer
        self.max_len=max_len
        self.label_list=label_list

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self.text_list)

    def __getitem__(self, index):
        """
        Get an item from the dataset.

        Args:
            index (int): Index of the item.

        Returns:
            tuple: Tuple containing the input_ids, attention_mask, and label.
        """
        text = self.text_list[index]
        label = self.label_list[index]
        tokenized = self.tokenizer(text=text,
                                   padding='max_length',
                                   truncation=True,
                                   max_length=self.max_len,
                                   return_tensors='pt')
        return tokenized['input_ids'].squeeze(), tokenized['attention_mask'].squeeze(), label