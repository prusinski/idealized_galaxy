import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
import yt
yt.enable_parallelism()
from yt.frontends.gizmo.api import GizmoDataset
from yt.units import dimensions
import sys
import h5py as h5
from scipy.signal import gaussian
from scipy.ndimage import filters
from radial_profile1 import find_center_iteratively
import trident

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

def read_amiga_rvir(amiga_data, output_fn, ds):
    """
    Figure out the virial radius from amiga's smoothed halo outputs
    (e.g., halo_0000_smooth.dat) for a given output_fn
    """
    output_number = int(os.path.basename(output_fn).split('.')[0][-3:])
    halo = amiga_data[:,0] == output_number
    rvir = amiga_data[halo,4][0]
    return ds.quan(rvir, 'code_length')

if __name__ == '__main__':

    fn_list = open(sys.argv[1], 'r')
    fns = fn_list.readlines()

    #amiga_data = get_amiga_data(sys.argv[2])
    #amiga_data = smooth_amiga(amiga_data)
    for fn in yt.parallel_objects(fns):
        fn = fn.strip()
        fn_head = fn.split('/')[-1]
        rprof_fn = "%s_rprof.h5" % fn_head
        ds = GizmoDataset(fn)
        trident.add_ion_fields('H') # add H I mass field
        c = find_center_iteratively(fn, ds=ds)
        #c = read_amiga_center(amiga_data, fn, ds)
        #rvir = read_rockstar_rvir(rockstar_data, ds)
        #_, c = ds.find_max('density')
        rvir = ds.quan(100, 'kpc')
        ds.unit_registry.add('r_vir', rvir.to("cm").tolist(), dimensions.length, tex_repr=r"r_{vir}")
        radial_extent = 4
        sp = ds.sphere(c, radial_extent*rvir)
        bulk_vel = sp.quantities.bulk_velocity()
        sp.set_field_parameter("bulk_velocity", bulk_vel)
    
        rp1 = yt.create_profile(sp, ('gas', 'radius'), ('gas', 'radial_velocity'),
                                weight_field=('gas', 'H_p0_mass'),
                                #weight_field=('gas', 'mass'),
                                #units = {('gas', 'radius'): 'r_vir'},
                                units = {('gas', 'radius'): 'kpc'},
                                logs = {('gas', 'radius'): False}, n_bins=128)
    
        rprof_file = h5.File(rprof_fn, 'w')
        rprof_file.create_dataset("radius", data=rp1.x.value)
        rprof_file.create_dataset("radial_velocity", data=rp1['radial_velocity'].to('km/s').value)
        rprof_file.close()
