import numpy as np
from astropy.io import fits
from astropy import units as u
from pathlib import Path
import ccdproc as ccdp
from ccdproc import ImageFileCollection
from math import cos
from math import radians



def tograd(t):
    return t[0]*15 + t[1]/60*15 + t[2]/3600*15

raw_files_path = Path('result')                                     # путь (название папки) к интересующим кадрам
raw_images = ccdp.ImageFileCollection(raw_files_path / 'G')
# raw_images.summary['file', 'imagetyp', 'filter', 'exptime', 'naxis1', 'naxis2']   # таблица данных об изображениях(для тестирования)
file_names = raw_images.files                                       # Список с наименованием файлов [list of str]
Mid_coords = open('AirMass.txt', 'w')
for i in range(len(file_names)):                                    # Цикл в котором: 1) в шапку записываются необходимые данные 2) меняется формат из .fit -> .fits
    data, header = fits.getdata(raw_files_path / 'G' / file_names[i], header=True)
    z = 1/cos(radians(90 - float(header['OBJCTALT'])))
    Mid_coords.write(str(z) + '\n')







'''
# coordinates on Earth
lat, long = 43.6781, 41.4597
lat, long = lat * u.deg, long * u.deg
altitude = 2070 * u.m

# location of obervations
obs_location = EarthLocation.from_geodetic(lat=lat, lon=long, height=altitude)

# observation time
# указывается время на 3ч раньше чем указано в шапке наблюдения)
obs_time = Time(datetime(2024, 4, 8, 18, 10, 16),scale='utc',location=obs_location)

# RIGHT ASCENSION AND DECLINATION at zenith for given lat, long
right_ascension_z = obs_time.sidereal_time('mean').to('deg')
declination_z = lat

print(right_ascension_z)
print(declination_z)
'''
print('debug')