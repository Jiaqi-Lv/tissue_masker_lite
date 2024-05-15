import os

import click
import numpy as np

from tissue_masker_lite.process.process_wsi import gen_tissue_mask

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "model_weights/model_22.pth"
)


@click.command()
@click.option("--wsi_path", type=click.Path(path_type=str))
@click.option("--save_dir", type=click.Path(path_type=str), default="./output")
@click.option(
    "--model_weight",
    type=click.Path(path_type=str),
    default="model_weights/model_36.pth",
)
@click.option("--threshold", type=float, default=0.5)
@click.option("--device", type=str, default="cuda")
@click.option("--return_mask/--no-return_mask", default=True)
@click.option("--save_mask/--no-save_mask", default=True)
def _get_mask(
    wsi_path: str,
    save_dir: str,
    model_weight: str,
    threshold: float,
    device: str,
    return_mask: bool,
    save_mask: bool,
):
    """
    Generate tissue mask of the input WSI

    Args:
        wsi_path(str): path to the input WSI
        save_dir(str): directory to save the output mask
        model_weight(str): path to the pre-trained model weight
        threshold(float): binary mask threshold (range between 0.0-1.0), default=0.5
        device(str): device to run the model on, options are "cuda", "mps", "cpu"
        return_mask(bool): Whether to return the output mask
        save_mask(bool): Whether to save the output mask
    Returns:
        mask(np.ndarray): returns tissue mask if return_mask is True
    """
    return get_mask(
        wsi_path=wsi_path,
        save_dir=save_dir,
        model_weight_path=model_weight,
        threshold=threshold,
        device=device,
        return_mask=return_mask,
        save_mask=save_mask,
    )


def get_mask(
    wsi_path: str,
    save_dir: str,
    model_weight: str = DEFAULT_MODEL_PATH,
    threshold: float = 0.5,
    device: str = "cuda",
    return_mask: bool = True,
    save_mask: bool = True,
) -> np.ndarray:
    """
    Generate tissue mask of the input WSI

    Args:
        wsi_path(str): path to the input WSI
        save_dir(str): directory to save the output mask
        model_weight(str): path to the pre-trained model weight
        threshold(float): binary mask threshold (range between 0.0-1.0), default=0.5
        device(str): device to run the model on, options are "cuda", "mps", "cpu"
        return_mask(bool): Whether to return output mask, default=True
        save_mask(bool): Whether to save output mask, default=True
    Returns:
        mask(np.ndarray): returns tissue mask if return_mask is True
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    return gen_tissue_mask(
        wsi_path=wsi_path,
        save_dir=save_dir,
        model_weight_path=model_weight,
        threshold=threshold,
        device=device,
        return_mask=return_mask,
        save_mask=save_mask,
    )


if __name__ == "__main__":
    _get_mask()
