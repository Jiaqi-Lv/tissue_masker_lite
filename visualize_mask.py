import matplotlib.pyplot as plt
import numpy as np

mask_path = "/home/u1910100/GitHub/tissue_masker_lite/output/TCGA-CJ-4912-01Z-00-DX1.28805E39-F07A-40FF-8810-162BED671E17.npy"
mask = np.load(mask_path)
plt.imshow(mask)
plt.show()
