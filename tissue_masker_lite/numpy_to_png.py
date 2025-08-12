import os
import pickle

import cv2
import numpy as np
from tiatoolbox.utils.misc import imwrite
from tiatoolbox.wsicore.wsireader import WSIReader
from tqdm import tqdm

wsi_dir = "/home/u1910100/web-public/tcga"
masks_dir = "/home/u1910100/lab-share/jiaqi/tcga_masks_qc"
save_dir = "/home/u1910100/lab-share/jiaqi/tcga_masks_qc"

for mask in tqdm(os.listdir(masks_dir)):
    ext = os.path.splitext(mask)[1]
    if ext != ".npy":
        continue

    mask_path = os.path.join(masks_dir, mask)
    wsi_name = os.path.splitext(mask)[0]
    wsi_path = os.path.join(wsi_dir, f"{wsi_name}.svs")
    rgb_mask_path = os.path.join(save_dir, f"{wsi_name}_mask.png")
    slide_thumb_path = os.path.join(save_dir, f"{wsi_name}_thumb.png")

    if os.path.exists(rgb_mask_path) and os.path.exists(slide_thumb_path):
        continue

    wsi_reader = WSIReader.open(wsi_path)
    slide_thumb = wsi_reader.slide_thumbnail(resolution=1.25, units="power")

    imwrite(slide_thumb_path, slide_thumb)

    binary_mask = np.load(mask_path)
    rgb_mask = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2RGB) * 255
    imwrite(rgb_mask_path, rgb_mask)
