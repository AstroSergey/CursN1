from photutils.datasets import load_star_image
from astropy.stats import sigma_clipped_stats
from astropy.nddata import Cutout2D
from astropy.wcs import WCS

from twirl import find_peaks

import numpy as np
import matplotlib.pyplot as plt
from photutils.aperture import CircularAperture

from astropy.wcs.utils import proj_plane_pixel_scales

from twirl import gaia_radecs
from twirl.geometry import sparsify

from twirl import compute_wcs


from pathlib import Path
from astropy.io import fits


'''
hdu = load_star_image()
data, true_wcs = hdu.data, WCS(hdu.header)
mean, median, std = sigma_clipped_stats(data, sigma=3.0)
'''
hdu = fits.open(Path('B+s') / 'a_Bfilt_00001.fits')
data = hdu[0].data
header = hdu[0].header
true_wcs = WCS(header)
mean, median, std = sigma_clipped_stats(data, sigma=3.0)

xy = find_peaks(data)[0:20]


'''
plt.imshow(data, vmin=np.median(data), vmax=3 * np.median(data), cmap="Greys_r")
_ = CircularAperture(xy, r=10.0).plot(color="y")
plt.show()
'''


fov = (data.shape * proj_plane_pixel_scales(true_wcs))[0]

center = true_wcs.pixel_to_world(*np.array(data.shape) / 2)

all_radecs = gaia_radecs(center, fov)

# we only keep stars 0.01 degree apart from each other
all_radecs = sparsify(all_radecs, 0.01)

wcs = compute_wcs(xy, all_radecs[0:30], tolerance=10)

radecs_xy = np.array(wcs.world_to_pixel_values(all_radecs))
plt.imshow(data, vmin=np.median(data), vmax=3 * np.median(data), cmap="Greys_r")
_ = CircularAperture(radecs_xy, 5).plot(color="y", alpha=0.5)

plt.shpow()



print('debug')