import os
import copy
import uuid
import torch
import random
import yaml
import numpy as np

from vista.models import get_model
from vista.utils.logger import get_logger

OUT_FOLDER = "out"

def set_seed(seed):
    """Set the seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)


def run(parameters, run_name=None, log_params=True, log_on_file=True):
    set_seed(parameters.get("seed", 42))

    if run_name is None:
        run_name = str(uuid.uuid4())[:8]
        run_name = os.path.join(OUT_FOLDER, run_name)
        os.makedirs(run_name, exist_ok=True)

    params_filename = run_name + "/params.yaml"
    if log_params:
        with open(params_filename, "w") as f:
            yaml.dump(parameters, f)

    log_filename = run_name + "/log.log" if log_on_file else None
    logger = get_logger("Run", log_filename)
    logger.info("parameters:")
    logger.info(parameters)

    model = get_model(parameters["model"])

    if hasattr(model, "set_classes") and "classes" in parameters:
        model.set_classes(parameters["classes"])
    if hasattr(model, "fuse"):
        model.fuse()

    if "train" in parameters:
        model.train(save_dir=run_name, **parameters["train"])

    if "val" in parameters:
        model.val(save_dir=run_name, **parameters["val"])


evaluate = run  # backwards-compat alias
    
