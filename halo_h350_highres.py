"""
Generates impact parameter versus radial velocity plots along z-axis for FIRE simulations
using yt-4.0 branch of yt. Fast.
"""
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
import yt
yt.enable_parallelism()
import numpy as np
from yt.frontends.gizmo.api import GizmoDataset
import sys
from scipy.signal import gaussian
from scipy.ndimage import filters
import os
import cmocean
from radial_profile1 import find_center_iteratively
from astropy.io import fits, ascii
from astropy.table import Table

# using smoothing length as an analog for the size of the gas region
def _H_I_number(field, data):
    return data[('gas', 'H_p0_number_density')] * (4./3. * 3.14 * data[('gas', 'smoothing_length')]**1)

# This is to correct a problem in the tip of yt-4 that messes up the units
# of cylindrical radius to be a factor of kpc/cm too low
# def _cylindrical_radius_kpc(field, data):
#     return data[('PartType0', 'cylindrical_radius')] / (3e21)

yt.add_field(("gas","H_I_column_density"), function=_H_I_number, units="cm**(-2)", sampling_type='particle')
# yt.add_field(("PartType0","cylindrical_radius_kpc"), function=_cylindrical_radius_kpc, units="cm", sampling_type='particle')

# def get_amiga_data(fn):
#     """
#     Try to read in amiga data if data file provided;
#     """
#     # have to do a little fanciness here since amiga datafiles have variable
#     # numbers of columns, and genfromtxt wants fixed number of columns.
#     # so avoiding genfromtxt altogether and doing by hand
#     try:
#         amiga_fn = fn
#         amiga = open(amiga_fn, 'r')
#         centroid_list = []
#         # first line is text
#         amiga.readline()
#         for line in amiga:
#             centroid = line.split()
#             centroid_list.append(np.array(centroid, dtype=np.float64)[[0,7,8,9,13]])
#         amiga.close()
#         return np.array(centroid_list)
#     except IndexError:
#         amiga_data = None
#
# def smooth_amiga(arr, width=20, std=10):
#     """
#     Apply a guassian filter of width 20 to the halo centroid information to
#     get rid of jumps in the centroid position vs time
#     """
#     b = gaussian(width,std)
#     newarr = np.empty_like(arr)
#     newarr[:,0] = arr[:,0] # snapshot number in the same
#     for i in range(1,5):
#         newarr[:,i] = filters.convolve1d(arr[:,i], b/b.sum())
#     return newarr
#
# def read_amiga_center(amiga_data, output_fn, ds):
#     """
#     Figure out the halo center from amiga's smoothed halo outputs
#     (e.g., halo_0000_smooth.dat) for a given output_fn
#     """
#     output_number = int(os.path.basename(output_fn).split('.')[0][-3:])
#     halo = amiga_data[:,0] == output_number
#     center = amiga_data[halo,1:4][0]
#     return ds.arr(center, 'code_length')

def save_as_fits(phaseplot, filename, field=('gas', 'H_I_column_density')):
    hdu1 = fits.PrimaryHDU(phaseplot._profile[field].v.T)
    hdu1.writeto(filename, overwrite=1)

def save_as_text(phaseplot, filename):
    # f = open(filename, 'w')
    # f.write('impact parameter bin edges [kpc]\n')
    # for j in phaseplot._profile.x_bins.v: f.write('%f\n' % j)
    # f.write('radial velocity bin edges [km/s]\n')
    # for j in phaseplot._profile.y_bins.v: f.write('%f\n' % j)
    # f.close()

    bs = phaseplot._profile.x.v
    vs = phaseplot._profile.y.v
    btab = Table([bs], names=('b'))
    vtab = Table([vs], names = ('v'))
    ascii.write(btab, f'{filename}_b.txt', overwrite=True)
    ascii.write(vtab, f'{filename}_v.txt', overwrite=True)

if __name__ == '__main__':

    with open(sys.argv[1]) as f:
        fns = [fn.rstrip() for fn in f.readlines()]

    #amiga_data = get_amiga_data(sys.argv[2])
    #amiga_data = smooth_amiga(amiga_data)
    # axes = [[1,0,0], [0,1,0], [0,0,1]]
    axes = 2*np.random.rand(30,3)-1

    for fn in yt.parallel_objects(fns):
        fn = fn.strip()
        fn_head = fn.split('/')[-1]
        num = int(fn_head.split('.')[0][-3:])
        ds = GizmoDataset(fn)
        #ds = yt.load(fn)
        c = find_center_iteratively(fn, ds=ds)
        #c = read_amiga_center(amiga_data, fn, ds)
        #_, c = ds.find_max('density')
        rvir = ds.quan(30, 'kpc')
        sp = ds.sphere(c, rvir)
        bulk_vel = sp.quantities.bulk_velocity()
        #print("Bulk Velocity of Halo = %s" % bulk_vel.to('km/s'))
        sp.set_field_parameter("bulk_velocity", bulk_vel)
        for i, ax in enumerate(axes):
            ad = ds.all_data()
            ad.set_field_parameter('center', c)
            ad.set_field_parameter('bulk_velocity', bulk_vel)
            ad.set_field_parameter('normal', np.array(ax))
            #p = yt.PhasePlot(ad, ('PartType0', 'cylindrical_radius_kpc'), ('PartType0', 'velocity_cylindrical_z'), ('gas', 'H_I_number'), weight_field=('gas', 'ones'))
            p = yt.PhasePlot(ad, ('PartType0', 'cylindrical_radius'), ('PartType0', 'velocity_cylindrical_z'), ('gas', 'H_I_column_density'), weight_field=None)
            p.set_unit(('PartType0', 'cylindrical_radius'), 'kpc')
            p.set_unit(('PartType0', 'velocity_cylindrical_z'), 'km/s')
            p.set_log(('PartType0', 'velocity_cylindrical_z'), False)
            p.set_log(('PartType0', 'cylindrical_radius'), False)
            p.set_xlim(1e1, 250)
            p.set_ylim(-1500,1500)
            p.set_cmap(('gas', 'H_I_column_density'), cmocean.cm.thermal)
            #p.set_zlim(('gas', 'H_I_number'), 1e12, 1e25)
            p.set_xlabel('Impact Parameter (kpc)')
            p.set_ylabel('Line of Sight Velocity (km/s)')
            p.set_title(('gas', 'H_I_column_density'), f"$\hat{{n}} = \\langle {ax[0]:.2f}, {ax[1]:.2f}, {ax[2]:.2f} \\rangle$")
            fn = 'images/KBSS_%03i_%1i' % (num, i)
            p.save(fn+'.png')
            save_as_fits(p, fn+'.fits')
            if i == 0:
                save_as_text(p, fn)
