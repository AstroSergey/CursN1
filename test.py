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

img1 = fits.open(dir / 'SA104(3)-001B.fits')[0].data

target = img1

fig = plt.figure(figsize=[10,8])

plt.imshow(target, cmap = 'gray', norm = LogNorm())
cbar = plt.colorbar()
plt.show()

print('debug')

