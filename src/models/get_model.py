def get_model(model_name, env,config):
    """Get the model based on the model name."""
    if model_name == "dqn":
        from models.dqn import Agent
        return Agent
    else:
        raise ValueError(f"Model {model_name} not recognized.")