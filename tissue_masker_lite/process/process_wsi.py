import os
import warnings
from pprint import pprint

import numpy as np
import segmentation_models_pytorch as smp
import torch
from tiatoolbox.models.engine.semantic_segmentor import SemanticSegmentor
from tiatoolbox.tools.patchextraction import get_patch_extractor
from tiatoolbox.wsicore.wsireader import WSIReader
from tqdm.auto import tqdm

from tissue_masker_lite.utils.helpers import (imagenet_normalise,
                                              morpholoy_post_process)

warnings.filterwarnings("ignore")


def pred_wsi(
    wsi_path: str, model: torch.nn.Module, device: str = "cuda"
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Predict an WSI

    Args:
        wsi_path(str): path to WSI
        model(torch.nn.Module): model
        device(str): device to run predictions on, options are "cuda", "mps", "gpu"
    Returns:
        predictions(list[np.ndarray]): a list of patch prediction output
        coordinates(list[np.ndarray]): a list of bounding boxes for each patch
    """
    reader = WSIReader.open(wsi_path)

    patch_extractor = get_patch_extractor(
        input_img=reader,
        method_name="slidingwindow",
        patch_size=(512, 512),
        stride=(480, 480),
        resolution=1.25,
        units="power",
    )

    predictions = []

    torch_device = torch.device(device)

    for patch in tqdm(patch_extractor, leave=False):
        patch = (patch / 255).astype(np.float32)

        input_patch = imagenet_normalise(patch)
        input_patch = np.moveaxis(input_patch, 2, 0)
        input_patch = torch.from_numpy(input_patch)
        input_patch = torch.unsqueeze(input_patch, 0)

        if device == "mps":
            input_patch = input_patch.to(torch.float32).to(torch_device)
        else:
            input_patch = input_patch.to(torch_device).float()

        with torch.no_grad():
            pred = model(input_patch)
            pred = torch.nn.functional.sigmoid(pred)
            pred = pred.detach().cpu().numpy()
            pred_mask = pred[0]

        pred_mask = np.squeeze(pred_mask, 0)

        predictions.append(pred_mask)

    return predictions, patch_extractor.coordinate_list


def gen_tissue_mask(
    wsi_path: str,
    save_dir: str,
    model_weight_path: str = "model_weights/model_36.pth",
    threshold: float = 0.5,
    device: str = "cuda",
    return_mask=True,
    save_mask=True,
) -> np.ndarray | None:
    """
    Generate tissue mask for an WSI

    Args:
        wsi_path(str): path to WSI
        save_dir(str): directory to save the tissue mask in
        model_weight_path(str): path to the pre-trained model weights
        threshold(float): binary mask threshold (range between 0.0-1.0), default=0.5
        cuda(bool): Whether to use CUDA
        device(str): device to run the model on, options are "cuda", "mps", "cpu"
        return_mask(bool): Whether to return output mask
        save_mask(bool): Whether to save output mask
    Returns:
        mask(np.ndarray): returns tissue mask if return_mask is True
    """
    fn = os.path.basename(wsi_path)
    pprint(f"Processing {fn}")
    fn = os.path.splitext(fn)[0]

    model = smp.Unet(
        encoder_name="efficientnet-b0",
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )

    pprint(f"Loading model to {device}")
    torch_device = torch.device(device)
    model.load_state_dict(torch.load(model_weight_path, map_location=torch_device))
    model.to(torch_device)
    model.eval()

    reader = WSIReader.open(wsi_path)

    dimension = reader.slide_dimensions(resolution=1.25, units="power")

    predictions, coordinates = pred_wsi(wsi_path, model, device=device)

    mask = SemanticSegmentor.merge_prediction(
        (dimension[1], dimension[0]), predictions, coordinates
    )

    mask = np.where(mask > threshold, 1, 0)
    mask = mask.astype(np.uint8)
    mask = morpholoy_post_process(mask[:, :, 0])
    save_path = os.path.join(save_dir, f"{fn}.npy")
    if save_mask:
        np.save(save_path, mask)
        pprint(f"mask saved at: {save_path}")
    pprint("Task finished successfully")
    if return_mask:
        return mask
