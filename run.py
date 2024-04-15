import os
from pprint import pprint

import click

from process.process_wsi import gen_tissue_mask


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
def run(wsi_path: str, save_dir: str, model_weight: str, threshold: str, cuda: bool):
    pprint(f"Input WSI: {wsi_path}")
    pprint(f"Output mask location: {save_dir}")
    pprint(f"{cuda}")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    gen_tissue_mask(
        wsi_path=wsi_path,
        save_dir=save_dir,
        model_weight_path=model_weight,
        threshold=threshold,
        return_mask=False,
    )


if __name__ == "__main__":
    run()
