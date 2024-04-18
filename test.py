import segmentation_models_pytorch as smp
from pprint import pprint
import torch
import numpy as np

model = smp.Unet(
        encoder_name="efficientnet-b0",
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )
model.to('cuda')

model.load_state_dict(torch.load("/home/u1910100/GitHub/tissue_masker_lite/tissue_masker_lite/model_weights/model_36.pth"))

pprint(model)
