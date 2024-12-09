import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from datetime import datetime
from astropy.coordinates import EarthLocation, SkyCoord, AltAz
from astropy.time import Time
from astropy.io import fits
from astropy import units as u

def tograd(t):
    return t[0]*15 + t[1]/60*15 + t[2]/3600*15

file = open('standarts.txt', 'r')
lines = file.readlines()
file.close()

Cat = []
for i in range(2, len(lines)):
    Cat.append(lines[i].split(';'))

# ra   dec   name   V   eV   N   ra   dec

Ra, Dec = [], []
for i in range(len(Cat)):
    if int(Cat[i][5]) > 10:
        Ra.append(float(Cat[i][0]))
        Dec.append(float(Cat[i][1]))
'''
StartCordsRa = [[11, 44, 54]]
StartCordsDec = [43+48.8/60]
EndCordsRa = [[17, 46, 26]]
EndCordsDec = [43+42.3/60]
'''

#day:       1-2     2-3     3-4     4-5     5-6     6-7     7-8     8-9     9-10
# start:    day     day     day     day     day     day     day     day     day
# end:      1:34    2:31    3:18    3:55    morning morning morning morning morning
# [2024, 4, day, 20, 25] сумерки - начало
# [2024, 4, day, 4, 0] сумерки - конец
Start = [[2024, 3, 31, 20, 25], [2024, 4, 1, 20, 25], [2024, 4, 2, 20, 25]]
End = [[2024, 4, 1, 1, 34], [2024, 4, 2, 2, 31], [2024, 4, 3, 3, 18]]
StartCordsRa = [[8, 51, 28], [8, 55, 25], [8, 59, 22], [9, 3, 19], [9, 7, 16], [9, 11, 13], [9, 15, 10]]
StartCordsDec = [43+51/60, 43+51.17/60, 43+51.28/60, 43+51.38/60, 43+51.47/60, 43+51.58/60, 43+51.67/60]
EndCordsRa = [[14, 59, 13], [15, 3, 10], [15, 54, 18], [16, 35, 24], [16, 44, 21], [16, 48, 18], [16, 52, 14]]
EndCordsDec = [43+46/60, 43+51.4/60, 43+50/60, 43+48.72/60, 43+48/60, 43+47.87/60, 43+47.73/60]
# https://stackoverflow.com/questions/75794481/astropy-zenith-verification-calculation-lat-long-time-to-zenith-and-vice-vers
# coordinates on Earth
lat, long = 43.6781, 41.4597
lat, long = lat * u.deg, long * u.deg
altitude = 2070 * u.m

# location of obervations
obs_location = EarthLocation.from_geodetic(lat=lat, lon=long, height=altitude)

# observation time
obs_time = Time(datetime(2024, 3, 31, 20, 25),scale='utc',location=obs_location)

# RIGHT ASCENSION AND DECLINATION at zenith for given lat, long
right_ascension_z = obs_time.sidereal_time('mean', long).to('deg')
declination_z = lat

print(right_ascension_z)
print(declination_z)

'''
NewSTandarts = open('NewStandarts.txt', 'w')
for i in range(len(Cat)):
    if int(Cat[i][5]) > 10:
        NewSTandarts.write(f'{Cat[i][0]}; {Cat[i][1]}; {Cat[i][2]}; {Cat[i][3]}\n')
NewSTandarts.close()
'''
for i in range(len(StartCordsRa)):
    c1 = plt.Circle((tograd(StartCordsRa[i]), StartCordsDec[i]), radius=80, color='r', alpha= .3)
    c2 = plt.Circle((tograd(EndCordsRa[i]), EndCordsDec[i]), radius=80, color='#1111FF', alpha= .3)
    plt.gca().add_artist(c1)
    plt.gca().add_artist(c2)
    plt.plot(Ra, Dec, '.k')
    plt.show()







print('debug')
