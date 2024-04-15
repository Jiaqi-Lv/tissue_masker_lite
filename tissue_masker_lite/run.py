import os
from pprint import pprint
import numpy as np
import click
from tissue_masker_lite.process.process_wsi import gen_tissue_mask


@click.command()
@click.option("--wsi_path", type=click.Path(path_type=str))
@click.option("--save_dir", type=click.Path(path_type=str), default="./output")
@click.option(
    "--model_weight",
    type=click.Path(path_type=str),
    default="model_weights/model_36.pth",
)
@click.option("--threshold", type=float, default=0.5)
@click.option("--cuda/--no-cuda", default=True)
@click.option("--return_mask/--no-return_mask", default=True)
@click.option("--save_mask/--no-save_mask", default=True)
def _get_mask(
    wsi_path: str,
    save_dir: str,
    model_weight: str,
    threshold: float,
    cuda: bool,
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
        cuda(bool): Whether to use CUDA
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
        cuda=cuda,
        return_mask=return_mask,
        save_mask=save_mask
    )


def get_mask(
    wsi_path: str,
    save_dir: str,
    model_weight: str,
    threshold: float = 0.5,
    cuda: bool = True,
    return_mask: bool = True,
    save_mask: bool = True
) -> np.ndarray:
    """
    Generate tissue mask of the input WSI

    Args:
        wsi_path(str): path to the input WSI
        save_dir(str): directory to save the output mask
        model_weight(str): path to the pre-trained model weight
        threshold(float): binary mask threshold (range between 0.0-1.0), default=0.5
        cuda(bool): Whether to use CUDA
        return_mask(bool): Whether to return output mask
        save_mask(bool): Whether to save output mask
    Returns:
        mask(np.ndarray): returns tissue mask if return_mask is True
    """
    pprint(f"Input WSI: {wsi_path}")
    pprint(f"Output mask location: {save_dir}")
    pprint(f"{cuda}")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    return gen_tissue_mask(
        wsi_path=wsi_path,
        save_dir=save_dir,
        model_weight_path=model_weight,
        threshold=threshold,
        return_mask=return_mask,
        save_mask=save_mask
    )


if __name__ == "__main__":
    _get_mask()
