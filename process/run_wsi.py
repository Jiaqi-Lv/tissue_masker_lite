from utils.helpers import morpholoy_post_process, imagenet_normalise
import segmentation_models_pytorch as smp
import torch
from tiatoolbox.wsicore.wsireader import WSIReader
from tiatoolbox.tools.patchextraction import get_patch_extractor
from tiatoolbox.models.engine.semantic_segmentor import (
    SemanticSegmentor,
    IOSegmentorConfig,
)
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
import os


def pred_wsi(wsi_path, model, resolution=0, units="level"):
    reader = WSIReader.open(wsi_path)

    patch_extractor = get_patch_extractor(
        input_img=reader,
        method_name="slidingwindow",
        patch_size=(512, 512),
        stride=(128, 128),
        resolution=resolution,
        units=units,
    )

    predictions = []

    for patch in tqdm(patch_extractor, leave=False):
        patch = (patch / 255).astype(np.float32)

        input_patch = imagenet_normalise(patch)
        input_patch = np.moveaxis(input_patch, 2, 0)
        input_patch = torch.from_numpy(input_patch)
        input_patch = torch.unsqueeze(input_patch, 0)
        input_patch = input_patch.to("cuda").float()

        with torch.no_grad():
            pred = model(input_patch)
            pred = torch.nn.functional.sigmoid(pred)
            pred = pred.detach().cpu().numpy()
            pred_mask = pred[0]

        # pred_mask[pred_mask>=threshold] = 1
        # pred_mask[pred_mask<threshold] = 0
        # pred_mask.astype(int)

        pred_mask = np.squeeze(pred_mask, 0)

        predictions.append(pred_mask)

    return predictions, patch_extractor.coordinate_list


def gen_tissue_mask(
    wsi_path, save_dir, model_weight_path="model_weights/model_22.pth", gpu=True
):
    
    fn = os.path.basename(wsi_path)
    fn = os.path.splitext(fn)[0]

    model = smp.Unet(
        encoder_name="efficientnet-b0",
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )

    if gpu:
        model.to("cuda")
    model.load_state_dict(torch.load(model_weight_path))
    model.eval()

    reader = WSIReader.open(wsi_path)

    dimension = reader.slide_dimensions(resolution=1.25, units="power")

    predictions, coordinates = pred_wsi(wsi_path, model, resolution=1.25, units="power")

    mask = SemanticSegmentor.merge_prediction(
        (dimension[1], dimension[0]), predictions, coordinates
    )
    threshold = 0.5
    mask = np.where(mask > threshold, 1, 0)
    mask = morpholoy_post_process(mask)
    mask = mask.astype(int)
    save_path = os.path.join(save_dir, f"{fn}.png")
    plt.imshow(mask, cmap="grey")
    plt.axis("off")
    plt.savefig(
        save_path,
        transparent=True,
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close()
    pprint("mask saved")