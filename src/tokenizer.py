from transformers import AutoTokenizer



def get_tokenizer(config_model):
    """
    Returns a tokenizer object based on the provided model configuration.

    Args:
        config_model (dict): A dictionary containing the model configuration.

    Returns:
        tokenizer: A tokenizer object.

    """
    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config_model['model_path'])
    return tokenizer