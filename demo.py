import argparse
from typing import Callable, Tuple
import torch 
import torch.nn.functional as F 
import numpy as np 
from tqdm import tqdm 

from copy import deepcopy
from trainer import ConservativeObjectiveTrainer
from model import SimpleMLP
from metrics import spearman_correlation
from utils import (
    build_data_loader,
    set_seed
)

tkwargs = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "dtype": torch.float32,
}

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=2024)
parser.add_argument("--num-solutions", type=int, default=1)
args = parser.parse_args() 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

_TKWARGS = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "dtype": torch.float32
}

def eval_fn(x: np.ndarray) -> np.ndarray:
    assert x.shape[1] == 2
    return x.sum(axis=1)

def generate_data(func: Callable, num_data: int, num_dim: int) \
    -> Tuple[np.ndarray, np.ndarray]:
    x = np.random.rand(num_data, num_dim)
    y = func(x)
    return x, y

x_mean, x_std, y_mean, y_std = None, None, None, None 

def normalize_x(x: np.ndarray) -> np.ndarray:
    global x_mean, x_std 
    if x_mean is None or x_std is None:
        x_mean = x.mean(axis=0) 
        x_std = x.std(axis=0) 
    return (x - x_mean) / x_std

def normalize_y(y: np.ndarray) -> np.ndarray:
    global y_mean, y_std 
    if y_mean is None or y_std is None:
        y_mean = y.mean() 
        y_std = y.std() 
    return (y - y_mean) / y_std

def denormalize_x(x: np.ndarray) -> np.ndarray:
    global x_mean, x_std 
    assert x_mean is not None and x_std is not None 
    return x * x_std + x_mean 

def denormalize_y(y: np.ndarray) -> np.ndarray:
    global y_mean, y_std 
    assert y_mean is not None and y_std is not None 
    return y * y_std + y_mean 



def run(config):
    x_data, y_data = generate_data(eval_fn, 1000, 2)
    print(x_data.shape, y_data.shape)

    x_data = normalize_x(x_data)
    y_data = normalize_y(y_data)
    
    config["input_shape"] = x_data[0].shape
    
    model = SimpleMLP(x_data.shape[1], 
                      hidden_dim=[2048, 2048], 
                      output_dim=1).to(**tkwargs)
    
    train_loader, val_loader = build_data_loader(
        x_data, y_data, batch_size=128, 
        require_valid=True, valid_ratio_if_valid=0.2,
        drop_last=False
    )
    
    trainer = ConservativeObjectiveTrainer(
        model, 
        config,
    )
    
    trainer.launch(train_loader, val_loader)
    
    
if __name__ == "__main__":
    set_seed(args.seed)
    config = deepcopy(args.__dict__)
    config.update(
        {
            "forward_lr": 1e-3,
            "n_epochs": 50,
            "alpha": 0.1,
            "alpha_lr": 0.01,
            "overestimation_limit": 0.5,
            "particle_lr": 0.05,
            "particle_gradient_steps": 50,
            "entropy_coefficient": 0.0,
            "noise_std": 0.0
        }
    )
    run(config) 
    