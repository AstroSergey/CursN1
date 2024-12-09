import os

import numpy as np


from matplotlib import pyplot as plt
from matplotlib import rc

from astropy.visualization import hist
from astropy.stats import mad_std


# plt.style.use('guide.mplstyle')


# Set some default parameters for the plots below
rc('font', size=20)
rc('axes', grid=True)

# Set up the random number generator, allowing a seed to be set from the environment
seed = os.getenv('GUIDE_RANDOM_SEED', None)

if seed is not None:
    seed = int(seed)
    
# This is the generator to use for any image component which changes in each image, e.g. read noise
# or Poisson error
noise_rng = np.random.default_rng(seed)

n_distributions = 100
bias_level = 1000
n_side = 320
bits = noise_rng.normal(size=(n_distributions, n_side**2)) + bias_level
average = np.average(bits, axis=0)
median = np.median(bits, axis=0)

fig, ax = plt.subplots(1, 2, sharey=True, tight_layout=True, figsize=(20, 10))

hist(bits[0, :], bins='freedman', ax=ax[0]);
ax[0].set_title('Одно распределение')
ax[0].set_xlabel('Значение в пикселе')

hist(average, bins='freedman', label='среднее', alpha=0.5, ax=ax[1]);
hist(median, bins='freedman', label='медиана', alpha=0.5, ax=ax[1]);
ax[1].set_title('{} Распределений'.format(n_distributions))
ax[0].set_ylabel('Кол-во пикселей')
ax[1].set_xlabel('Значение в пикселе')
ax[1].legend()

plt.show()
print('debug')








