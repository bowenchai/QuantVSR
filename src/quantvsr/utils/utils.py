import torch

def recursive_to(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {key: recursive_to(value, device) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [recursive_to(elem, device) for elem in obj]
    elif isinstance(obj, tuple):
        return tuple(recursive_to(elem, device) for elem in obj)
    else:
        return obj
