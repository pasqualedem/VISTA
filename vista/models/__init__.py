from .moondream import MoonDream
from .omdetturbo import OmDetTurbo
from .sam.src.sam3_model import Sam3Model

MODEL_ZOO = {
    "moondream": MoonDream,
    "omdetturbo": OmDetTurbo,
    "sam": Sam3Model,
}


def get_model(parameters: dict) -> object:
    model_name = parameters.get("name")
    kwargs = {k: v for k, v in parameters.items() if k != "name"}
    if model_name not in MODEL_ZOO:
        raise ValueError(f"Model '{model_name}' not found in MODEL_ZOO. Available models: {list(MODEL_ZOO.keys())}")
    return MODEL_ZOO[model_name](**kwargs)