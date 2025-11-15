from typing import Dict, Type, Callable
from .base import BaseVSRModel

_MODEL_REGISTRY: Dict[str, Type[BaseVSRModel]] = {}


def register_model(name: str):
    def decorator(cls: Type[BaseVSRModel]):
        if name in _MODEL_REGISTRY:
            raise ValueError(f"Model '{name}' already registered!")
        _MODEL_REGISTRY[name] = cls
        return cls

    return decorator


def get_model(name: str, **kwargs) -> BaseVSRModel:
    if name not in _MODEL_REGISTRY:
        raise ValueError(f"Model '{name}' not found. Available: {list(_MODEL_REGISTRY.keys())}")
    return _MODEL_REGISTRY[name](**kwargs)


def list_models() -> list:
    return list(_MODEL_REGISTRY.keys())
