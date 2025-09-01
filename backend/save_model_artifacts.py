import torch
import os
import json

def save_model_artifacts(model: torch.nn.Module, path_to_save: str, config: dict = None) -> None:
    """
    Saves the trained COMET transformer model and optional config to disk.

    Args:
        model (torch.nn.Module): Trained PyTorch model to save.
        path_to_save (str): Directory path where model artifacts will be stored.
        config (dict, optional): Model configuration dictionary to save as JSON.

    Outputs:
        None – Saves model weights and config to disk.
    """
    # Ensure the save directory exists
    os.makedirs(path_to_save, exist_ok=True)

    # Save model weights
    weights_path = os.path.join(path_to_save, "comet_model_weights.pt")
    torch.save(model.state_dict(), weights_path)
    print(f" Model weights saved to: {weights_path}")

    # Save model config if provided
    if config:
        config_path = os.path.join(path_to_save, "comet_model_config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)
        print(f" Model config saved to: {config_path}")

