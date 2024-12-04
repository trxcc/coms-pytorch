import torch 
import random 
import numpy as np 

from typing import Optional, Union, Tuple
from torch.utils.data import DataLoader, TensorDataset, random_split

_TKWARGS = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "dtype": torch.float32
}

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.determinstic = True
    
def build_data_loader(
    x: Union[np.ndarray, torch.Tensor], 
    y: Union[np.ndarray, torch.Tensor],
    batch_size: int = 128,
    require_valid: bool = True,
    valid_ratio_if_valid: float = 0.2,
    drop_last: bool = False
) -> Tuple[DataLoader, Optional[DataLoader]]:
    
    # assert len(x.shape) == 2, "X should be of shape (#data, input_size)"
    assert (not require_valid) or 0 <= valid_ratio_if_valid <= 1, \
            "valid ratio should be within [0, 1]"
    assert batch_size >= 1
    
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).to(**_TKWARGS)
    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y).to(**_TKWARGS)
    
    train_dataset = TensorDataset(x, y)
    data_size = len(train_dataset)
    
    if require_valid:
        lengths = [
            data_size - int(valid_ratio_if_valid * data_size),
            int(valid_ratio_if_valid * data_size)
        ]
        train_dataset, validate_dataset = random_split(train_dataset, lengths)
        
    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        drop_last=drop_last
    )
    
    valid_loader = DataLoader(
        dataset=validate_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=drop_last
    ) if require_valid else None
    
    return (
        train_loader,
        valid_loader if require_valid else None
    )