from training import get_model_for_deployment

def get_model(cfg, weights_path=None):
    """
    Returns a model object based on the provided model configuration.

    Args:
        cfg (dict): A dictionary containing the model configuration.

    Returns:
        model: A model object.

    """
    # model
    if weights_path is None:
        model = get_model_for_deployment(cfg['model_path'], cfg['fine_tuned_model_path'])
    else:
        model = get_model_for_deployment(cfg['model_path'], weights_path)
    return model