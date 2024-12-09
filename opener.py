from astropy.io import fits

from pathlib import Path




hdu = fits.open('image-radec.fits')

data = hdu[1].data
header = hdu[1].header
table = open('quer.txt', 'w')
for i in range(len(data)):
        table.write(str(data[i][0]) + '\t' + str(data[i][1]) + '\n')
'''
    table.write(str(data[i][0]) + '\t' + str(data[i][1]) + '\t'+ str(data[i][2]) + '\t'+ str(data[i][3]) + '\t'+ str(data[i][4]) + 
                '\t'+ str(data[i][5]) + '\n')
    
'''
table.close
print('debug')
