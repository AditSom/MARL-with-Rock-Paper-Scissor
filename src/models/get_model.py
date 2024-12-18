def get_model(model_name, env, config):
    """
    Get the model based on the model name.
    
    Parameters:
    model_name (str): The name of the model to retrieve.
    env: The environment in which the model will be used.
    config: Configuration settings for the model.
    
    Returns:
    class: The class of the model requested.
    
    Raises:
    ValueError: If the model_name is not recognized.
    """
    
    # Check if the model name is "dqn"
    if model_name == "dqn":
        # Import the Agent class from the dqn module
        from models.dqn import Agent
        # Return the Agent class
        return Agent
    else:
        # Raise an error if the model name is not recognized
        raise ValueError(f"Model {model_name} not recognized.")