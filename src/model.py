from training import get_model_for_deployment

def get_model(cfg):
    """
    Returns a model object based on the provided model configuration.

    Args:
        cfg (dict): A dictionary containing the model configuration.

    Returns:
        model: A model object.

    """
    # model
    model = get_model_for_deployment(cfg['model_path'], cfg['fine_tuned_model_path'])
    return model