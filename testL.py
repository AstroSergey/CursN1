from pathlib import Path
from astropy.io import fits
from astropy.nddata import CCDData
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np


import os
import ccdproc as ccdp
from ccdproc import ImageFileCollection

import astroalign as aa


dir = Path('result')              
imgs = ccdp.ImageFileCollection(dir)    
img_list = imgs.files         
n = 10
i = 13
ref_data = CCDData.read(dir / img_list[n])
target = (fits.open(dir / img_list[n])[0].data)
data_to_align = fits.open(dir / img_list[i])[0].data
print(img_list[n])
print(img_list[i])
source = data_to_align

img1 = fits.open(dir/ img_list[i])[0].data
img0 = fits.open(dir / img_list[n])[0].data


img_aligned, footprint = aa.register(source, target, detection_sigma=5)


fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes[0, 0].imshow(source, cmap='gray', norm = LogNorm(), interpolation='none', origin='lower')
axes[0, 0].axis('off')
axes[0, 0].set_title("Source Image")

axes[0, 1].imshow(target, cmap='gray', norm = LogNorm(), interpolation='none', origin='lower')
axes[0, 1].axis('off')
axes[0, 1].set_title("Target Image")
'''
axes[1, 0].imshow(img_aligned, cmap='gray', interpolation='none', origin='lower')
axes[1, 0].axis('off')
axes[1, 0].set_title("Source Image aligned with Target")

axes[1, 1].imshow(footprint, cmap='gray', interpolation='none', origin='lower')
axes[1, 1].axis('off')
axes[1, 1].set_title("Footprint of the transformation")

axes[1, 0].axis('off')
'''
plt.tight_layout()
plt.show()

print('debug')