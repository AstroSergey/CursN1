import numpy as np
from photutils.aperture import SkyCircularAperture, CircularAnnulus, ApertureStats
from photutils.aperture import aperture_photometry
from astropy.io import fits



from photutils.aperture import CircularAperture
positions = [(1967.0, 1348), (1047, 1112)]
aperture = CircularAperture(positions, r=25.0)

hdu = fits.open('test.fits')
data = hdu[0].data
hdu.close()
phot_table = aperture_photometry(data, aperture)
print(phot_table)


# Define an annular aperture for background estimation
annulus_r_in = 35.0 # Inner radius of the annulus
annulus_r_out = 45.0 # Outer radius of the annulus
annuli = CircularAnnulus(positions, r_in=annulus_r_in, r_out=annulus_r_out)

# Perform photometry for the annulus
bkg_table = aperture_photometry(data, annuli)

# Calculate the mean background per pixel
annulus_area = annuli.area
bkg_mean = bkg_table['aperture_sum'] / annulus_area

# Subtract background from the aperture flux
phot_table['residual_sum'] = phot_table['aperture_sum'] - bkg_mean * aperture.area
print(phot_table)

stat = ApertureStats(data, aperture)

'''
import matplotlib.pyplot as plt
from astropy.visualization import simple_norm

# Display the image
norm = simple_norm(data, 'sqrt', percent=99)
plt.imshow(data, norm=norm, cmap='Greys')
plt.colorbar()

# Overlay the apertures
aperture.plot(color='blue', lw=1.5, alpha=0.7)
annuli.plot(color='red', lw=1.5, alpha=0.5)

plt.show()
'''
# Сигнал с вычетом фона
signal = phot_table['residual_sum']

# Фоновый шум
background_noise = np.sqrt(bkg_mean * aperture.area)

# Общий шум (если известен шум считывания)
readout_noise = 15.0                                                        # Задайте шум считывания вашей камеры (если известно)
total_noise = np.sqrt(signal + background_noise**2 + readout_noise**2)

# Отношение сигнал/шум (SNR)
snr = signal / total_noise

# Добавляем результат в таблицу
phot_table['signal'] = signal
phot_table['snr'] = snr

print(phot_table)



