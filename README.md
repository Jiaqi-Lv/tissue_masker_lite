# tissue_masker_lite
This repository contains a deep learning based approach to creating tissue masks from whole-slide images.

# How to use
## 1. Install it as a Python Package locally
Run `pip install .`  
## 2. Example usage:  
```
from tissue_masker_lite import get_mask
wsi_path = "path/to/wsi"
save_dir = "path/to/save_dir"
model_weight_path = "tissue_masker_lite/model_weights/model_22.pth"
mask = get_mask(
    wsi_path=input_wsi,
    save_dir=save_dir,
    model_weight=model_weight_path,
    threshold=0.5,
    cuda=True,
    return_mask=True
    )
```
