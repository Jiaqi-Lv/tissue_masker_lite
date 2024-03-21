import click
from pprint import pprint
from pathlib import Path
from model import run_wsi


@click.command()
@click.argument("wsi_path", type=click.Path(exists=True, readable=True, path_type=str))
@click.option("--save_dir", type=click.Path(exists=False, path_type=str), default="./")
def run(wsi_path, save_dir):
    pprint(f"Input WSI: {wsi_path}")
    pprint(f"Output mask location: {save_dir}")


if __name__ == "__main__":
    run()
