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
from multiprocessing import Pool
import trident
from functools import partial
from itertools import repeat

# using smoothing length as an analog for the size of the gas region
def _H_I_number(field, data):
    return data[('gas', 'H_p0_number_density')] * data[('gas', 'smoothing_length')]

def _CII_col(field, data):
    return data[('gas', 'C_p1_number_density')] * data[('gas', 'smoothing_length')]

def _CIV_col(field, data):
    return data[('gas', 'C_p3_number_density')] * data[('gas', 'smoothing_length')]

def _O_col(field, data):
    return data[('gas', 'O_p0_number_density')] * data[('gas', 'smoothing_length')]

def _SiII_col(field, data):
    return data[('gas', 'Si_p1_number_density')] * data[('gas', 'smoothing_length')]

def _SiIV_col(field, data):
    return data[('gas', 'Si_p3_number_density')] * data[('gas', 'smoothing_length')]

# This is to correct a problem in the tip of yt-4 that messes up the units
# of cylindrical radius to be a factor of kpc/cm too low
# def _cylindrical_radius_kpc(field, data):
#     return data[('PartType0', 'cylindrical_radius')] / (3e21)

yt.add_field(("gas","H_I_column_density"), function=_H_I_number, units="cm**(-2)", sampling_type='particle')

# yt.add_field(("PartType0","cylindrical_radius_kpc"), function=_cylindrical_radius_kpc, units="cm", sampling_type='particle')




def save_as_text(phaseplot, filename):

    bs = phaseplot._profile.x.v
    vs = phaseplot._profile.y.v
    btab = Table([bs], names=('b'))
    vtab = Table([vs], names = ('v'))
    ascii.write(btab, f'{filename}_b.txt', overwrite=True)
    ascii.write(vtab, f'{filename}_v.txt', overwrite=True)


quants = [("gas","H_I_column_density"),
            ("gas","C_II_column_density"),
            ("gas","C_IV_column_density"),
            ("gas","O_I_column_density"),
            ("gas","Si_II_column_density"),
            ("gas","Si_IV_column_density")]

def hmap(pencili, pencil, num, dsin):

    ad = dsin.all_data()
    ad.set_field_parameter('center', c)
    ad.set_field_parameter('bulk_velocity', bulk_vel)
    ad.set_field_parameter('normal', np.array(pencil))

    p = yt.PhasePlot(ad, ('PartType0', 'cylindrical_radius'), ('PartType0', 'velocity_cylindrical_z'), quants, weight_field=None)
    p.set_unit(('PartType0', 'cylindrical_radius'), 'kpc')
    p.set_unit(('PartType0', 'velocity_cylindrical_z'), 'km/s')
    p.set_log(('PartType0', 'velocity_cylindrical_z'), False)
    p.set_log(('PartType0', 'cylindrical_radius'), False)
    p.set_xlim(1e1, 250)
    p.set_ylim(-1500,1500)
    p.set_cmap('all', cmocean.cm.thermal)
    #p.set_zlim(('gas', 'H_I_number'), 1e12, 1e25)
    p.set_xlabel('Impact Parameter (kpc)')
    p.set_ylabel('Line of Sight Velocity (km/s)')

    if yt.is_root():

        if pencili == 0:
            save_as_text(p, f'KBSS_{num}')

        for q in quants:
            p.set_title(q, f"$\hat{{n}} = \\langle {pencil[0]:.2f}, {pencil[1]:.2f}, {pencil[2]:.2f} \\rangle$")
            el = q[1].split("_")[0] + q[1].split("_")[1]
            p[q].save(f'images/KBSS_{num}_{el}_{pencili}.png')
            fits.PrimaryHDU(p._profile[q].v.T).writeto(f'images/KBSS_{num}_{el}_{pencili}.fits', overwrite=1)




if __name__ == '__main__':

    with open(sys.argv[1]) as f:
        fns = [fn.rstrip() for fn in f.readlines()]
    #amiga_data = get_amiga_data(sys.argv[2])
    #amiga_data = smooth_amiga(amiga_data)
    # axes = [[1,0,0], [0,1,0], [0,0,1]]
    axes = 2*np.random.rand(30,3)-1

    # for fn in yt.parallel_objects(fns):
    fn = fns[0]
    fn_head = fn.split('/')[-1]
    n = int(fn_head.split('.')[0][-3:])

    ds = GizmoDataset(fn)
    trident.add_ion_fields(ds, ions=['C IV', 'Si II', 'O I', 'C II', 'Si IV'])

    ds.add_field(("gas","C_II_column_density"), function=_CII_col, units="cm**(-2)", sampling_type='particle')
    ds.add_field(("gas","C_IV_column_density"), function=_CIV_col, units="cm**(-2)", sampling_type='particle')
    ds.add_field(("gas","O_I_column_density"), function=_O_col, units="cm**(-2)", sampling_type='particle')
    ds.add_field(("gas","Si_II_column_density"), function=_SiII_col, units="cm**(-2)", sampling_type='particle')
    ds.add_field(("gas","Si_IV_column_density"), function=_SiIV_col, units="cm**(-2)", sampling_type='particle')

    #ds = yt.load(fn)
    c = find_center_iteratively(fn, ds=ds)
    #c = read_amiga_center(amiga_data, fn, ds)
    #_, c = ds.find_max('density')
    rvir = ds.quan(30, 'kpc')
    sp = ds.sphere(c, rvir)
    bulk_vel = sp.quantities.bulk_velocity()
    #print("Bulk Velocity of Halo = %s" % bulk_vel.to('km/s'))
    sp.set_field_parameter("bulk_velocity", bulk_vel)


    # print(product(enumerate(axes),))
    # print(zip(list(range(len(axes))), axes, repeat(n), repeat(ds)))
    for i, a in enumerate(axes):
        hmap(i,a, n, ds)
    # pool = Pool()
    # pool.starmap(hmap, zip(list(range(len(axes))), axes, repeat(n), repeat(ds)))
