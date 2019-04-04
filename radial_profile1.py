"""
Make x,y,z column density projections for a dataset, centered on the largest
halo in the simulation.  Specify the files you want to run this on using a 
filelist where each line of the list is the path to the file.

FIRE: 
    ibrun python radial_profile1.py filelist.txt halo_00000_smooth.dat

Tempest:
    aprun python radial_profile1.py filelist.txt halo_5016/tree_51.dat 51

"""
import yt
yt.enable_parallelism()
import numpy as np
import trident
import h5py as h5
import sys
import glob
import os.path
from yt.units.yt_array import \
    YTArray, \
    YTQuantity
from yt.frontends.gizmo.api import GizmoDataset
import cmocean
from scipy.signal import filtfilt, gaussian
from scipy.ndimage import filters
import ytree

def log(string):
    """
    Log information to STDOUT
    """
    length = len(string)
    equals = "="*length
    print(equals)
    print(string)
    print(equals)

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

def find_center(ds):
    """
    Find the halo center for a zoom sim by finding highest star density point
    """
    log("Finding the galaxy center")
    #c = ds.find_max(('gas', 'density'))
    #return c[1]
    _, c1 = ds.find_max(('deposit', 'PartType4_density'))
    return c

def distance(point, coords):
    """
    Calculate the distances between a point and a bunch of coordinates
    """
    dists = np.array([(point[ax]-coords[:,ax])**2 for ax in range(3)])
    dists = np.sqrt(np.sum(dists, axis=0))
    return dists

def center(coords, mass):
    """
    Find the center of mass of a bunch of coordinates and masses
    """
    center = np.array([(coords[:,ax] * mass).sum() for ax in range(3)])
    center /= mass.sum()
    return center

def create_mask(point, coords, frac=0.9):
    """
    Create a mask to mask out points in the outer 25% distance from a point
    """
    radius = distance(point, coords)
    rad_max = np.max(radius)
    print("   radius = %f" % rad_max)
    mask = radius <= frac*rad_max
    return mask, rad_max

def find_center_iteratively(fn, frac=0.9, n_parts=50, ds=None):
    """
    Find the center of mass for a bunch of FIRE particles iteratively
    """
    f = h5.File(fn, 'r')
    coords = f['PartType4/Coordinates'][()]
    mass = f['PartType4/Masses'][()]

    # Iteratively figure out the center of mass
    for i in range(500):
        print("Pass %i: %i particles" % (i, len(mass)))
        c = center(coords, mass)
        if coords.shape[0] <= n_parts:
            break
        mask, rad_max = create_mask(c, coords, frac)
        coords = coords[mask]
        mass = mass[mask]
    f.close()
    #print(c)
    if ds is not None:
        c = ds.arr(c, 'code_length')
    #print(c)
    return c

def make_phase(ds, data_source):
    """
    """
    ph = yt.PhasePlot(data_source, 'density', 'temperature', ['mass'], weight_field=None)
    ph.set_log('density', True)
    ph.set_unit('density', 'g/cm**3')
    ph.set_log('temperature', True)
    ph.set_unit('temperature', 'K')
    ph.set_xlim(1e-30,1e-20)
    ph.set_ylim(1e1,1e7)
    ph.set_cmap('mass', 'dusk')
    ph.set_unit('mass', 'Msun')
    ph.set_zlim('mass', 5e4,5e8)
    ph.save('images/')

def make_projection(ds, axis, ion_fields, center, width, data_source):
    """
    Use QuadProj and FRBs to make projection
    """
    p = ds.proj(ion_fields, axis, weight_field=None, center=center, 
                data_source=data_source, method='integrate')
    return p.to_frb(width, 800, center=center)

def make_projection_old(ds, axis, ion_fields, center, width, data_source, radius=None, weight_field=None):
    """
    Use ProjectionPlot to make projection (cannot specify resolution)
    """
    p = yt.ProjectionPlot(ds, axis, ion_fields, center=center, width=width, 
                          data_source=data_source, weight_field=weight_field)
    p.hide_axes()
    p.annotate_scale()
    p.annotate_timestamp(redshift=True)
    if radius:
        r = radius.in_units('kpc')
        p.annotate_sphere(center, (r, 'kpc'), circle_args={'color':'white', 'alpha':0.5, 'linestyle':'dashed', 'linewidth':5})
    for field in ion_fields:
        p.set_cmap(field, 'dusk')
        set_image_details(p, field, True)
        p.set_background_color(field)
    p.save('images/')
    return p.frb

def make_off_axis_projection(ds, vec, north_vec, ion_fields, center, width, data_source, radius, weight_field=None, dir=None):
    """
    Use OffAxisProjectionPlot to make projection (cannot specify resolution)
    """
    p = yt.OffAxisProjectionPlot(ds, vec, ion_fields, center=center, width=width, 
                                 data_source=data_source, north_vector=north_vec, weight_field=weight_field)
    p.hide_axes()
    p.annotate_scale()
    p.annotate_timestamp(redshift=True)
    r = radius.in_units('kpc')
    p.annotate_sphere(center, (r, 'kpc'), circle_args={'color':'white', 'alpha':0.5, 'linestyle':'dashed', 'linewidth':5})
    for field in ion_fields:
        p.set_cmap(field, 'dusk')
        set_image_details(p, field, True)
        p.set_background_color(field)
    if dir is None:
        dir = 'face/'
    p.save(os.path.join('images', dir))
    return p.frb

def set_image_details(plot, field, projection=True):
    """
    Set appropriate cmap and zlim and label
    """
    #if projection:
    p = plot
    if 1:
        if field == ('gas', 'density'):
            p.set_cmap(field, 'dusk')
            p.set_zlim(field, 1e-5, 1e1)
        if field == ('gas', 'H_number_density'):
            p.set_cmap(field, 'thermal')
            p.set_zlim(field, 1e13, 1e21)
            p.set_colorbar_label(field, "H I Column Density (cm$^{-2}$)")
        if field == ('gas', 'Mg_p1_number_density'):
            p.set_cmap(field, 'haline')
            #p.set_zlim(field, 1e5, 1e18)
            p.set_colorbar_label(field, "Mg II Column Density (cm$^{-2}$)")
        if field == ('gas', 'C_p1_number_density'):
            p.set_cmap(field, 'haline')
            #p.set_zlim(field, 1e10, 1e18)
            p.set_colorbar_label(field, "C II Column Density (cm$^{-2}$)")
        if field == ('gas', 'C_p2_number_density'):
            p.set_cmap(field, 'haline')
            #p.set_zlim(field, 1e10, 1e18)
            p.set_colorbar_label(field, "C III Column Density (cm$^{-2}$)")
        if field == ('gas', 'Si_p1_number_density'):
            p.set_cmap(field, 'haline')
            #p.set_zlim(field, 1e5, 1e18)
            p.set_colorbar_label(field, "Si II Column Density (cm$^{-2}$)")
        if field == ('gas', 'Si_p2_number_density'):
            p.set_cmap(field, 'haline')
            #p.set_zlim(field, 1e5, 1e18)
            p.set_colorbar_label(field, "Si III Column Density (cm$^{-2}$)")
        if field == ('gas', 'Si_p3_number_density'):
            p.set_cmap(field, 'haline')
            #p.set_zlim(field, 1e8, 1e16)
            p.set_colorbar_label(field, "Si IV Column Density (cm$^{-2}$)")
        if field == ('gas', 'N_p1_number_density'):
            p.set_cmap(field, 'haline')
            #p.set_zlim(field, 1e7, 1e19)
            p.set_colorbar_label(field, "N II Column Density (cm$^{-2}$)")
        if field == ('gas', 'N_p2_number_density'):
            p.set_cmap(field, 'haline')
            #p.set_zlim(field, 1e9, 1e17)
            p.set_colorbar_label(field, "N III Column Density (cm$^{-2}$)")
        if field == ('gas', 'N_p4_number_density'):
            p.set_cmap(field, 'haline')
            #p.set_zlim(field, 1e11, 1e17)
            p.set_colorbar_label(field, "N V Column Density (cm$^{-2}$)")
        if field == ('gas', 'O_p5_number_density'):
            p.set_cmap(field, 'haline')
            p.set_zlim(field, 1e13, 1e16)
            p.set_colorbar_label(field, "O VI Column Density (cm$^{-2}$)")
        if field == ('gas', 'Ne_p7_number_density'):
            p.set_cmap(field, 'haline')
            #p.set_zlim(field, 1e13, 1e16)
            p.set_colorbar_label(field, "Ne VIII Column Density (cm$^{-2}$)")
        if field == ('gas', 'temperature'):
            p.set_cmap(field, 'solar')
            p.set_zlim(field, 1e4, 1e7)
        if field == ('gas', 'O_nuclei_density'):
            #p.set_zlim(field, 1e14, 1e22)
            p.set_colorbar_label(field, "Oxygen Column Density (cm$^{-2}$)")
            p.set_cmap(field, 'ice')
        if field == ('gas', 'metallicity'):
            p.set_cmap(field, 'ice')
            #p.set_zlim(field, 1e-3, 1e-1)
        if field == ('gas', 'metal_density'):
            p.set_cmap(field, 'ice')
            p.set_zlim(field, 1e-8, 1e0)
            p.set_colorbar_label(field, "Projected Metal Density (g cm$^{-2}$)")

def get_rockstar_data(rstar_fn, halo_id):
    """
    Use ytree to get all of the halo centroids, virial radii, and redshift info; store in a dict
    """
    # load up dataset and get appropriate TreeNode
    a = ytree.load(rstar_fn)
    t = a[a["Orig_halo_ID"] == halo_id][0]

    redshift_arr = t['prog', 'redshift']
    x_arr = t['prog', 'x'].in_units('unitary')
    y_arr = t['prog', 'y'].in_units('unitary')
    z_arr = t['prog', 'z'].in_units('unitary')
    rvir_arr = t['prog', 'Rvir'].convert_to_units('kpc') 

    return {'redshift_arr':redshift_arr, 'x_arr':x_arr, 'y_arr':y_arr, 'z_arr':z_arr, 'rvir_arr':rvir_arr}

def read_rockstar_center(rockstar_data, ds):
    """
    Interpolate halo center from rockstar merger tree
    """
    redshift = ds.current_redshift
    redshift_arr = rockstar_data['redshift_arr']
    x = np.interp(redshift, redshift_arr, rockstar_data['x_arr'].in_units('unitary'))
    y = np.interp(redshift, redshift_arr, rockstar_data['y_arr'].in_units('unitary'))
    z = np.interp(redshift, redshift_arr, rockstar_data['z_arr'].in_units('unitary'))

    # Construct YTArray with correct units of original dataset (i.e., unitary)
    #arr = np.array([x,y,z]) * rockstar_data['x_arr'][0].uq
    arr = ds.arr([x,y,z], 'unitary')
    return arr
    
def read_rockstar_rvir(rockstar_data, ds):
    """
    Interpolate halo virial radius from rockstar merger tree
    """
    redshift = ds.current_redshift
    redshift_arr = rockstar_data['redshift_arr']
    rvir_arr = rockstar_data['rvir_arr']
    rvir = np.interp(redshift, redshift_arr, rvir_arr)
    rvir = ds.quan(rvir, 'kpccm').in_units('kpc')
    return rvir

if __name__ == '__main__':

    """
    Generate projections, phase diagrams, and column density hdf5 tables for 
    simulation snapshots.  filelist.txt simply lists the path of files
    to be processed, one per line.
    """
    AMIGA = False
    Rockstar = False
    plot_grid = False
    if len(sys.argv) == 2:
        pass
    elif len(sys.argv) == 3:
        AMIGA = True
    elif len(sys.argv) == 4:
        Rockstar = True
    else:
        print("""
        usage FIRE: 
        python radial_profile1.py filelist.txt [halo_00000_smooth.dat] 
        or Tempest:
        python radial_profile1.py filelist.txt halo_5016/tree_51.dat 51
        """)
        sys.exit()

    # Variables to set for each run
    radial_extent = YTQuantity(100, 'kpc') # kpc
    width = 2*radial_extent
    res = 800 # default resolution of PlotWindow images

    # Loading datasets
    fn_list = open(sys.argv[1], 'r')
    fns = fn_list.readlines()

    # FIRE with AMIGA centroids
    if AMIGA:
        # Read in halo information from amiga output if present
        amiga_data = get_amiga_data(sys.argv[2])
        # Smooth the data to remove jumps in centroid
        amiga_data = smooth_amiga(amiga_data)

    # Tempest with Rockstar centroids
    elif Rockstar:
        rockstar_data = get_rockstar_data(sys.argv[2], int(sys.argv[3]))

    for fn in yt.parallel_objects(fns):
        fn = fn.strip() # Get rid of trailing \n
        fn_head = fn.split('/')[-1]
        cdens_fn = "%s_cdens.h5" % fn_head

        # Define ions we care about
        ions = []
        ion_fields = []
        full_ion_fields = []
        ions.append('H I')
        ion_fields.append('H_number_density')
        full_ion_fields.append(('gas', 'H_number_density'))
        #ions.append('Mg II')
        #ion_fields.append('Mg_p1_number_density')
        #full_ion_fields.append(('gas', 'Mg_p1_number_density'))
        #ions.append('Si II')
        #ion_fields.append('Si_p1_number_density')
        #full_ion_fields.append(('gas', 'Si_p1_number_density'))
        #ions.append('Si III')
        #ion_fields.append('Si_p2_number_density')
        #full_ion_fields.append(('gas', 'Si_p2_number_density'))
        #ions.append('Si IV')
        #ion_fields.append('Si_p3_number_density')
        #full_ion_fields.append(('gas', 'Si_p3_number_density'))
        #ions.append('N II')
        #ion_fields.append('N_p1_number_density')
        #full_ion_fields.append(('gas', 'N_p1_number_density'))
        #ions.append('N III')
        #ion_fields.append('N_p2_number_density')
        #full_ion_fields.append(('gas', 'N_p2_number_density'))
        #ions.append('N V')
        #ion_fields.append('N_p4_number_density')
        #full_ion_fields.append(('gas', 'N_p4_number_density'))
        #ions.append('C II')
        #ion_fields.append('C_p1_number_density')
        #full_ion_fields.append(('gas', 'C_p1_number_density'))
        #ions.append('C III')
        #ion_fields.append('C_p2_number_density')
        #full_ion_fields.append(('gas', 'C_p2_number_density'))
        #ions.append('Ne VIII')
        #ion_fields.append('Ne_p7_number_density')
        #full_ion_fields.append(('gas', 'Ne_p7_number_density'))
        #ions.append('O VI')
        #ion_fields.append('O_p5_number_density')
        #full_ion_fields.append(('gas', 'O_p5_number_density'))
        n_fields = len(ion_fields)

        others = []
        other_fields = []
        full_other_fields = []
        #others.append('Temperature')
        #other_fields.append('temperature')
        #full_other_fields.append(('gas', 'temperature'))
    
        log("Starting projections for %s" % fn)
        if AMIGA:
            ds = GizmoDataset(fn)
        elif Rockstar:
            ds = yt.load(fn)
            # These vectors are precalculated to be the correct viewing angle for an edge on view
            E1 = ds.arr([0.40607303, 0., -0.91384063], 'code_length')
            L = ds.arr([0.83915456, 0.3959494, 0.37288562], 'code_length') # edge-on disk view at z ~ 0.2
            #vec1 = [L, -E1]
            #vec2 = [E1, L]
            vec1 = []
            vec2 = []
            dir = ['face/', 'edge/']
        else:
            ds = yt.load(fn)
        trident.add_ion_fields(ds, ions=ions, ftype='gas')

        #ions.append('O_nuclei_density')
        #ion_fields.append('O_nuclei_density')
        #full_ion_fields.append(('gas', 'O_nuclei_density'))
        #ions.append('density')
        #ion_fields.append('density')
        #full_ion_fields.append(('gas', 'density'))
        #ions.append('metal_density')
        #ion_fields.append('metal_density')
        #full_ion_fields.append(('gas', 'metal_density'))

        # Figure out centroid and r_vir info
        if AMIGA:
            log("Reading amiga center for halo in %s" % fn)
            c = read_amiga_center(amiga_data, fn, ds)
            rvir = read_amiga_rvir(amiga_data, fn, ds)
        elif Rockstar:
            log("Reading rockstar center for halo in %s" % fn)
            c = read_rockstar_center(rockstar_data, ds)
            rvir = read_rockstar_rvir(rockstar_data, ds)
        else:
            log("No amiga/rockstar data; Finding centroid iteratively for %s" % fn)
            c = find_center_iteratively(fn, ds=ds)
            rvir = ds.quan(10, 'kpc')

        sp = ds.sphere(c, 3*radial_extent)
    
        cdens_file = h5.File(cdens_fn, 'a')

        # Create box around galaxy so we're only sampling galaxy out to 1 Mpc
        one = ds.arr([.5, .5, .5], 'Mpc')
        box = ds.box(c-one, c+one)
    
        # Identify the radius from the center of each pixel (in sim units)
        px, py = np.mgrid[-width/2:width/2:res*1j, -width/2:width/2:res*1j]
        radius = (px**2.0 + py**2.0)**0.5
        if "radius" not in cdens_file.keys():
            cdens_file.create_dataset("radius", data=radius.ravel())
    
        # Repeat for each dimension
        for axis in ['x', 'y', 'z']:
    
            # Only do a Projection if it hasn't been filled yet
            #if ion_fields[0] not in cdens_file.keys() or axis not in cdens_file[ion_fields[0]].keys():
            log("Generating projection in %s" % axis)
            frb = make_projection_old(ds, axis, full_ion_fields, c, width, box, rvir)
            for i, ion_field in enumerate(ion_fields):
                dset = "%s/%s" % (ion_field, axis)
                if dset not in cdens_file.keys():
                    cdens_file.create_dataset(dset, data=frb[full_ion_fields[i]].ravel())
                    cdens_file.flush()
            frb = make_projection_old(ds, axis, full_other_fields, c, width, box, rvir, \
                                      weight_field=('gas', 'density'))
            for i, other_field in enumerate(other_fields):
                dset = "%s/%s" % (other_field, axis)
                if dset not in cdens_file.keys():
                    cdens_file.create_dataset(dset, data=frb[full_other_fields[i]].ravel())
                    cdens_file.flush()

            # Add grid level SlicePlot
            if plot_grid:
                p = yt.ProjectionPlot(ds, axis, ('index', 'grid_level'), center=c, width=width, data_source=box, method='mip')
                p.hide_axes()
                p.annotate_scale()
                p.annotate_timestamp(redshift=True)
                r = rvir.in_units('kpc')
                #p.annotate_sphere(center, (r, 'kpc'), circle_args={'color':'white', 'alpha':0.5, 'linestyle':'dashed', 'linewidth':5})
                p.set_cmap('grid_level', 'ice')
                p.set_log('grid_level', False)
                p.set_zlim('grid_level', 5,10)
                p.set_background_color('grid_level')
                p.save('images/')

        # If Tempest, include faceon, edgeon views.
        if Rockstar:
            for axis in range(len(vec1)):
                log("Generating off-axis projection in %s" % axis)
                frb = make_off_axis_projection(ds, vec1[axis], vec2[axis], full_ion_fields, \
                                               c, width, box, rvir, dir=dir[axis])
                for i, ion_field in enumerate(ion_fields):
                    dset = "%s/%s" % (ion_field, dir[axis])
                    if dset not in cdens_file.keys():
                        cdens_file.create_dataset(dset, data=frb[full_ion_fields[i]].ravel())
                        cdens_file.flush()
                frb = make_off_axis_projection(ds, vec1[axis], vec2[axis], full_other_fields, \
                                               c, width, box, rvir, \
                                               weight_field=('gas', 'density'), dir=dir[axis])
                for i, other_field in enumerate(other_fields):
                    dset = "%s/%s" % (ion_field, dir[axis])
                    if dset not in cdens_file.keys():
                        dset = "%s/%s" % (other_field, dir[axis])
                        cdens_file.create_dataset(dset, data=frb[full_other_fields[i]].ravel())
                    cdens_file.flush()
        cdens_file.close() 
        #log("Generating phase diagram for %s" % fn)
        #make_phase(ds, sp)
