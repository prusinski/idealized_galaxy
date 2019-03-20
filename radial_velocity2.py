import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
import yt
from yt.frontends.gizmo.api import GizmoDataset
from yt.units import dimensions
import sys
import h5py as h5
import numpy as np

if __name__ == '__main__':

    fn_list = open(sys.argv[1], 'r')
    fns = fn_list.readlines()

    r_list = []
    rv_list = []

    for fn in fns:
        fn = fn.strip()
        fn_head = fn.split('/')[-1]
        rprof_fn = "%s_rprof.h5" % fn_head
        rprof_file = h5.File(rprof_fn, 'r')
        
        r_list.append(rprof_file['radius'].value)
        rv_list.append(rprof_file['radial_velocity'].value)
        rprof_file.close()

    r_arr = np.stack(i for i in r_list)
    rv_arr = np.stack(i for i in rv_list)

    n_bins = r_arr.shape[1]
    rv_meds = np.zeros([n_bins, 3])
    for i in range(n_bins):
        rv_meds[i,0] = np.median(rv_arr[:,i])
        rv_meds[i,1] = np.percentile(rv_arr[:,i], 25)
        rv_meds[i,2] = np.percentile(rv_arr[:,i], 75)

    
    median_file = h5.File('median_file.h5', 'w')
    median_file.create_dataset('radius', data=r_arr[0])
    median_file.create_dataset('radial_velocity', data=rv_meds)
    median_file.close()
    
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(r_arr[0], rv_meds[:,0], color='blue', linewidth=3)
    plt.fill_between(r_arr[0], rv_meds[:,1], rv_meds[:,2], facecolor='blue', alpha=0.2)

    #ax.set_xlabel(r"$\mathrm{r\ (rvir)}$")
    #ax.set_xlim(0,4)
    ax.set_xlabel(r"$\mathrm{r\ (kpc)}$")
    ax.set_ylabel(r"$\mathrm{v_{radial}\ (km/s)}$")
    ax.set_xlim(0,400)

    fig.savefig("velocity.png")
