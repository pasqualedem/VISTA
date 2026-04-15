# from label_anything.models.lam import Lam 


from copy import deepcopy
from safetensors import safe_open
from safetensors.torch import save_file
import torch


FLOAT_PRECISIONS = {
    "fp32": torch.float32,
    "fp64": torch.float64,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def torch_dict_load(file_path):
    if file_path.endswith(".pth") or file_path.endswith(".pt") or file_path.endswith(".bin"):
        return torch.load(file_path)
    if file_path.endswith(".safetensors"):
        with safe_open(file_path, framework="pt") as f:
            d = {k: f.get_tensor(k) for k in f.keys()}
        return d
    raise ValueError("File extension not supported")


def torch_dict_save(data, file_path):
    if file_path.endswith(".pth") or file_path.endswith(".pt") or file_path.endswith(".bin"):
        torch.save(data, file_path)
    elif file_path.endswith(".safetensors"):
        save_file(data, file_path)
    else:
        raise ValueError("File extension not supported")


def substitute_values(x: torch.Tensor, values, unique=None):
    """
    Substitute values in a tensor with the given values
    :param x: the tensor
    :param unique: the unique values to substitute
    :param values: the values to substitute with
    :return: the tensor with the values substituted
    """
    if unique is None:
        unique = x.unique()
    lt = torch.full((unique.max() + 1,), -1, dtype=values.dtype, device=x.device)
    lt[unique] = values
    return lt[x]


def to_device(batch, device):
    if isinstance(batch, (list, tuple)):
        return [to_device(b, device) for b in batch]
    if isinstance(batch, dict):
        return {k: to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    return batch


def linearize_metrics(metrics, id2class=None):
    linearized = {}
    for k, v in metrics.items():
        if isinstance(v, dict):
            linearized.update(linearize_metrics(v, id2class))
        if isinstance(v, torch.Tensor):
            # Check if it has a single item
            if len(v.shape) == 0:
                linearized[k] = v.item()
            else:
                for i, elem in enumerate(v):
                    class_name = id2class[i] if id2class is not None else f"class_{i}"
                    linearized[f"{k}_{class_name}"] = elem
                linearized[k] = v.mean().item()
                linearized[f"{k}_fg"] = v[1:].mean().item()
    return linearized


def clone_input_dict(input_dict):
    return {k: v.clone() if isinstance(v, torch.Tensor) else deepcopy(v) for k, v in input_dict.items()}