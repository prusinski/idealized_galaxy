%matplotlib inline
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import os
from astropy.io import fits, ascii
from astropy.table import Table
# from pyds9 import *

num = 190

# d=DS9()

btab = ascii.read(f'images/KBSS_{num}_0_b.txt')
bvec = btab['b'].data

vtab = ascii.read(f'images/KBSS_{num}_0_v.txt')
vvec = vtab['v'].data

vbts = [fits.open(f'images/KBSS_{num}_{i}.fits')[0].data for i in range(29)]

print(bvec.shape, vvec.shape, vbts[0].shape)

avg_vbt = np.nanmean(vbts, axis=0)
med_vbt = np.nanmedian(vbts, axis=0)
std_vbt = np.nanstd(vbts, axis=0)

# print(std_vbt)

vcents = []

for bi, b in enumerate(bvec):
    # if bi == 10:
    # vel centroids
    spec = avg_vbt[:, bi]
    velcent = np.trapz(vvec * spec, x=vvec)/np.trapz(spec, x=vvec)
    vcents.append(velcent)


# d.set_np2arr(std_vbt)

fig, ax = plt.subplots(1,3, sharex=True, sharey=True, figsize = (14,8))
ax[0].imshow(avg_vbt, aspect='auto',
            # vmin = 1e13, vmax=1e20,
            norm=mpl.colors.LogNorm(vmin = 1e15, vmax=1e25), interpolation='nearest',
            origin='lower', extent = [bvec.min(), bvec.max(), vvec.min(), vvec.max()])

ax[1].imshow(med_vbt, aspect='auto',
            # vmin = 1e13, vmax=1e20,
            norm=mpl.colors.LogNorm(vmin = 1e15, vmax=1e25), interpolation='nearest',
            origin='lower', extent = [bvec.min(), bvec.max(), vvec.min(), vvec.max()])

std = ax[2].imshow(std_vbt, aspect='auto',
            # vmin = 1e13, vmax=1e20,,
            norm=mpl.colors.LogNorm(vmin = 1e15, vmax=1e25), interpolation='nearest',
            origin='lower', extent = [bvec.min(), bvec.max(), vvec.min(), vvec.max()])

ax[0].plot(bvec, vcents, 'r.-')
ax[1].plot(bvec, vcents, 'r.-')
ax[2].plot(bvec, vcents, 'r.-')

cb = fig.colorbar(std, ax=ax[2], label='$N_\mathrm{HI}$ (cm$^{-2}$)')

ax[0].set_title('Average')
ax[1].set_title('Median')
ax[2].set_title('Standard Deviation')

ax[0].axhline(0, c='k', ls='--', lw = 0.7)
ax[1].axhline(0, c='k', ls='--', lw = 0.7)
ax[2].axhline(0, c='k', ls='--', lw = 0.7)

fig.supxlabel('$b$ (kpc)')
fig.supylabel('$v_\mathrm{LOS}$ (km s$^{-1}$)')
plt.savefig('h350_vbt_30.pdf', bbox_inches='tight')
plt.show()
