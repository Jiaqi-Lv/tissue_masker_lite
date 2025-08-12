from tissue_masker_lite import get_mask
import os
from tqdm import tqdm
import pickle

wsi_dir = "/home/u1910100/web-public/tcga"
# masks_dir = "/mnt/it-services/lab-share/tcga_masks_qc"

# to_do = []
# masks = []

# for mask in os.listdir(masks_dir):
#     mask_name = os.path.splitext(mask)[0]
#     masks.append(mask_name)

# for wsi in os.listdir(wsi_dir):
#     name = os.path.splitext(wsi)[0]
#     if not name in masks:
#         to_do.append(name)

# print(len(to_do))
# with open('to_do.txt', 'w') as file:
#     for name in to_do:
#         file.write(f"{name}\n")

# with open('to_do.pk', 'wb') as file:
#     pickle.dump(to_do, file)


# wsi_dir = "/home/u1910100/web-public/cptac"
save_dir = "/home/u1910100/lab-share/jiaqi/tcga_masks_qc"
os.makedirs(save_dir,exist_ok=True)

# wsi_list = os.listdir(wsi_dir)
# wsi_list.sort()

# this_batch = wsi_list[4000:]

with open ('to_do.pk', 'rb') as fp:
    to_do = pickle.load(fp)



for wsi_name in tqdm(to_do):
    wsi_path = os.path.join(wsi_dir, f"{wsi_name}.svs")
    save_full_path = os.path.join(save_dir, f"{wsi_name}.npy")
    if not os.path.exists(save_full_path):
        try:
            get_mask(
                wsi_path,
                save_dir,
                return_mask=False
            )
        except Exception as err:
            print(err)
            continue