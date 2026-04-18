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


def evaluate(parameters, run_name=None, log_params=True, log_on_file=True):
    set_seed(parameters.get("seed", 42))

    if run_name is None:
        run_name = str(uuid.uuid4())[:8]
        run_name = os.path.join(OUT_FOLDER, run_name)
        os.makedirs(run_name, exist_ok=True)
    # model filename is log filename but with .pt instead of .log
    params_filename = run_name + "/params.yaml"
    if log_params:
        with open(params_filename, "w") as f:
            yaml.dump(parameters, f)

    log_filename = run_name + "/log.log" if log_on_file else None
    logger = get_logger("Eval", log_filename)
    logger.info("parameters:")
    logger.info(parameters)
    
    model_parameters = parameters["model"]
    model = get_model(model_parameters)
    
    if hasattr(model, "set_classes") and "classes" in parameters:
        model.set_classes(parameters["classes"])
    
    val_params = parameters.get("val", {})
    
    model.val(save_dir=run_name, **val_params)
    
