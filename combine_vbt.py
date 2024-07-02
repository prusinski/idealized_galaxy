%matplotlib inline
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import os
from astropy.io import fits, ascii
from astropy.table import Table
from pyds9 import *

num = 570

d=DS9()

btab = ascii.read(f'images/KBSS_{num}_0_b.txt')
bvec = btab['b'].data

vtab = ascii.read(f'images/KBSS_{num}_0_v.txt')
vvec = vtab['v'].data

vbts = [fits.open(f'images/KBSS_{num}_{i}.fits')[0].data for i in range(15)]

avg_vbt = np.nanmean(vbts, axis=0)
std_vbt = np.nanstd(vbts, axis=0)

# print(std_vbt)

d.set_np2arr(std_vbt)

fig, ax = plt.subplots(1,2, sharex=True, sharey=True, figsize = (10,8))
ax[0].imshow(avg_vbt, aspect='auto',
            # vmin = 1e13, vmax=1e20,
            norm=mpl.colors.LogNorm(vmin = 1e13, vmax=1e17), interpolation='nearest',
            origin='lower', extent = [bvec.min(), bvec.max(), vvec.min(), vvec.max()])
std = ax[1].imshow(std_vbt, aspect='auto',
            # vmin = 1e13, vmax=1e20,,
            norm=mpl.colors.LogNorm(vmin = 1e13, vmax=1e17), interpolation='nearest',
            origin='lower', extent = [bvec.min(), bvec.max(), vvec.min(), vvec.max()])

cb = fig.colorbar(std, ax=ax[1], label='$N_\mathrm{HI}$ (cm$^{-2}$)')

ax[0].set_title('Average')
ax[1].set_title('Standard Deviation')
fig.supxlabel('$b$ (kpc)')
fig.supylabel('$v_\mathrm{LOS}$ (km s$^{-1}$)')
plt.savefig('MW_halo_vbt.pdf', bbox_inches='tight')
plt.show()
