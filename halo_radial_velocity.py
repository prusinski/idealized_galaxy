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
import trident

# using smoothing length as an analog for the size of the gas region
def _column_density(field, data):
    return data[('gas', 'H_number_density')] * data[('gas', 'smoothing_length')]

# This is to correct a problem in the tip of yt-4 that messes up the units
# of cylindrical radius to be a factor of kpc/cm too low
def _cylindrical_radius_kpc(field, data):
    return data[('PartType0', 'cylindrical_radius')] / (3e21)

yt.add_field(("gas","H_I_column_density"), function=_column_density, units="cm**-2", sampling_type='particle')
yt.add_field(("PartType0","cylindrical_radius_kpc"), function=_cylindrical_radius_kpc, units="cm", sampling_type='particle')

def get_amiga_data(fn):
    """
    Try to read in amiga data if data file provided;
    """
    # have to do a little fanciness here since amiga datafiles have variable
    # numbers of columns, and genfromtxt wants fixed number of columns.
    # so avoiding genfromtxt altogether and doing by hand
    try:
        amiga_fn = fn
        amiga = open(amiga_fn, 'r')
        centroid_list = []
        # first line is text
        amiga.readline()
        for line in amiga:
            centroid = line.split()
            centroid_list.append(np.array(centroid, dtype=np.float64)[[0,7,8,9,13]])
        amiga.close()
        return np.array(centroid_list)
    except IndexError:
        amiga_data = None

def smooth_amiga(arr, width=20, std=10):
    """
    Apply a guassian filter of width 20 to the halo centroid information to
    get rid of jumps in the centroid position vs time
    """
    b = gaussian(width,std)
    newarr = np.empty_like(arr)
    newarr[:,0] = arr[:,0] # snapshot number in the same
    for i in range(1,5):
        newarr[:,i] = filters.convolve1d(arr[:,i], b/b.sum())
    return newarr

def read_amiga_center(amiga_data, output_fn, ds):
    """
    Figure out the halo center from amiga's smoothed halo outputs
    (e.g., halo_0000_smooth.dat) for a given output_fn
    """
    output_number = int(os.path.basename(output_fn).split('.')[0][-3:])
    halo = amiga_data[:,0] == output_number
    center = amiga_data[halo,1:4][0]
    return ds.arr(center, 'code_length')


if __name__ == '__main__':

    with open(sys.argv[1]) as f:
        fns = [fn.rstrip() for fn in f.readlines()]
    
    #amiga_data = get_amiga_data(sys.argv[2])
    #amiga_data = smooth_amiga(amiga_data)
    for fn in yt.parallel_objects(fns):
        fn = fn.strip()
        fn_head = fn.split('/')[-1]
        ds = GizmoDataset(fn)
        c = find_center_iteratively(fn, ds=ds)
        trident.add_ion_fields(ds, 'H')
        #c = read_amiga_center(amiga_data, fn, ds)
        #_, c = ds.find_max('density')
        rvir = ds.quan(50, 'kpc')
        sp = ds.sphere(c, rvir)
        bulk_vel = sp.quantities.bulk_velocity()
        #print("Bulk Velocity of Halo = %s" % bulk_vel.to('km/s'))
        sp.set_field_parameter("bulk_velocity", bulk_vel)
        ad = ds.all_data()
        ad.set_field_parameter('center', c)
        ad.set_field_parameter('normal', np.array([1,0,0]))
        ad.set_field_parameter('bulk_velocity', bulk_vel)
        p = yt.PhasePlot(ad, ('gas', 'radius'), ('gas', 'radial_velocity'), ('gas', 'H_p0_mass'), weight_field=('gas', 'ones'))
        p.set_unit(('gas', 'radius'), 'kpc')
        p.set_unit(('gas', 'radial_velocity'), 'km/s')
        p.set_unit(('gas', 'H_p0_mass'), 'Msun')
        #p.set_log(('gas', 'radius'), False)
        #p.set_log(('gas', 'radial_velocity'), False)
        p.set_xlim(1e1, 1e4)
        p.set_ylim(-1000,1000)
        p.set_cmap(('gas', 'H_p0_mass'), 'thermal')
        #p.set_zlim(('gas', 'H_p0_mass'), 1e12, 1e25)
        p.set_xlabel('Radius (kpc)')
        p.set_ylabel('Radial Velocity (km/s)')
        p.save()
